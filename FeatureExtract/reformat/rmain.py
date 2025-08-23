import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import DictAction, Config
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.append('.')
from models.models import PointTransformer
from models.pointnet2_utils import interpolating_points
from reformat.data.mvtec3d import get_data_loader
from utils.au_pro_util import calculate_au_pro
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
from utils.utils import KNNGaussianBlur
from utils.utils import get_coreset_idx_randomp
from utils.utils import set_seeds


class PointPatchCore(torch.nn.Module):
    def __init__(self, group_size, num_group, n_reweight):
        super().__init__()
        self.backbone = PointTransformer(group_size, num_group)
        self.backbone.load_model_from_ckpt("checkpoints/pointmae_pretrain.pth")
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))

        self.blur = KNNGaussianBlur(4)

        # memory bank
        self.memory_bank = torch.empty(0)
        self.mean = 0
        self.std = 1

        # test
        self.n_reweight = n_reweight

        # evaluation results
        self.image_preds = []
        self.image_labels = []
        self.pixel_preds = []
        self.pixel_labels = []
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.ins_id = 0

    def preprocess(self, organized_pc):
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()  # 224x224x3
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)  # Nx3
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]  # remove empty pts
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        return unorganized_pc_no_zeros, nonzero_indices  # Bx3xN2, N,

    def forward_with_cache(self, organized_pc, cache_path):
        if False: # cache_path.exists():
            feat = torch.load(cache_path).to(args.device)
        else:
            feat = self.forward(organized_pc)
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(feat.cpu(), cache_path)
        return feat

    def forward(self, organized_pc):
        """
        organized_pc: Nx3xHxW
        """
        with torch.no_grad():
            batch, _, height, width = organized_pc.shape
            unorganized_pc_no_zeros, nonzero_indices = self.preprocess(organized_pc)
            xyz = unorganized_pc_no_zeros.to(next(self.parameters()).device).contiguous()
            xyz_feature_maps, center, neighbor_idx, center_idx = self.backbone(xyz)  # BxDxG, BxGx3, BxGxGs, BxG

            # scatter back to img view
            interpolated_pc = interpolating_points(xyz, center.permute(0, 2, 1), xyz_feature_maps)  # BxDxN
            feat_dim = interpolated_pc.shape[1]
            xyz_patch = xyz_feature_maps
            xyz_patch_full = torch.zeros((1, feat_dim, height * width),  # hard code for batch size == 1
                                         dtype=xyz_patch.dtype, device=xyz_patch.device)
            xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

            xyz_patch_full_2d = xyz_patch_full.view(1, feat_dim, height, width)
            xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

            # xyz_patch = xyz_patch_full_resized.reshape(feat_dim, -1).T  # NxD
        return xyz_patch_full_resized
        # {
        #     'xyz_patch': xyz_patch,
        #     'xyz_patch_full_resized': xyz_patch_full_resized,
        # }

    def compute_s_s_map(self, patch_feat, feature_map_dims, mask, label):
        _, _, height, width = mask.shape
        patch_feat = (patch_feat - self.mean) / self.std
        dist = torch.cdist(patch_feat, self.memory_bank)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch_feat[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.memory_bank[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.memory_bank)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.memory_bank[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch_feat.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(height, width), mode='bilinear')
        s_map = self.blur(s_map.cpu())

        self.image_preds.append(s.cpu().numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.squeeze().numpy())
        self.gts.append(mask.squeeze().numpy())

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        return {
            'image_rocauc': image_rocauc,
            'pixel_rocauc': pixel_rocauc,
            'au_pro': au_pro,
        }


def run_coreset(memory_bank, ratio, eps):
    if ratio < 1:
        coreset_idx = get_coreset_idx_randomp(memory_bank, n=int(ratio * memory_bank.shape[0]), eps=eps)
        memory_bank = memory_bank[coreset_idx]
    return memory_bank


def main():
    set_seeds(args.seed)
    cache_dir = Path(cfg.cache_dir) / f'{cfg.model.group_size}_{cfg.model.num_group}_{args.seed}'
    cache_dir.mkdir(exist_ok=True, parents=True)
    from collections import defaultdict
    all_results = defaultdict(list)
    for class_name in cfg.classes:
        model = PointPatchCore(cfg.model.group_size, cfg.model.num_group, cfg.test.n_reweight)
        model.to(args.device)
        model.train()  # train or eval?

        # extract features
        cache_path = cache_dir / f'{class_name}_train_all.pth'
        if False: # cache_path.exists():
            print(f'Load from cache {cache_path}')
            memory_bank = torch.load(cache_path)
        else:
            memory_list = []
            train_loader = get_data_loader("train", class_name=class_name, img_size=cfg.image_size)
            for i_train, sample in enumerate(
                    tqdm(train_loader, desc=f'Extracting train features for class {class_name}')):
                fcp = (Path(cache_dir) / 'feat' / sample['rgb_path'][0].replace('/', '_')).with_suffix('.pth')
                feat = model.forward_with_cache(sample['resized_organized_pc'], fcp)
                feat = feat.reshape(feat.shape[1], -1).T
                memory_list.append(feat.cpu())  # save in cpu to avoid oom
                if i_train >= cfg.max_train_iter:
                    break
            memory_bank = torch.cat(memory_list, 0)
            torch.save(memory_bank, cache_path)
        # save here

        if cfg.do_normalization:
            model.mean = torch.mean(memory_bank)  # save mean std for testing
            model.std = torch.std(memory_bank)
        memory_bank = (memory_bank - model.mean) / model.std

        # coreset sampling
        cache_path = cache_dir / f'{class_name}_train_coreset.pth'
        if cache_path.exists():
            model.memory_bank = torch.load(cache_path)
        else:
            print(f'Running coreset on class {class_name}...')
            model.memory_bank = run_coreset(memory_bank, cfg.coreset.ratio, cfg.coreset.eps).to(args.device)
            torch.save(model.memory_bank, cache_path)

        # evaluation
        model.eval()
        test_loader = get_data_loader("test", class_name=class_name, img_size=cfg.image_size)
        for i_test, sample in enumerate(tqdm(test_loader, desc=f'Extracting test features for class {class_name}')):
            with torch.no_grad():
                fcp = (Path(cache_dir) / 'feat' / sample['rgb_path'][0].replace('/', '_')).with_suffix('.eval.pth')
                feat = model.forward_with_cache(sample['resized_organized_pc'], fcp)
                feat_size = feat.shape[-2:]
                feat = feat.reshape(feat.shape[1], -1).T
                model.compute_s_s_map(feat, feat_size, sample['gt_mask'], sample['label'])
                if i_test >= cfg.max_test_iter:
                    break
        ret = model.calculate_metrics()
        for k, v in ret.items():
            all_results[k].append(v)
        print(all_results)
    print(all_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--config', '-c', help='config file path.', default='reformat/configs/point.py')
    parser.add_argument("--device", default='cuda', type=str, choices=['cuda', 'cpu'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    main()

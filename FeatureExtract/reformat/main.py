import argparse

import numpy as np
import torch
from mmcv import DictAction, Config
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.models import PointTransformer
from models.pointnet2_utils import interpolating_points
from reformat.data.mvtec3d import get_data_loader
from utils.au_pro_util import calculate_au_pro
from utils.mvtec3d_util import organized_pc_to_unorganized_pc
from utils.utils import KNNGaussianBlur
from utils.utils import get_coreset_idx_randomp


class PointPatchCore(torch.nn.Module):
    def __init__(self, group_size, num_group):
        self.backbone = PointTransformer(group_size, num_group)
        self.backbone.load_model_from_ckpt("checkpoints/pointmae_pretrain.pth")
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))

        self.blur = KNNGaussianBlur(4)

        # memory bank
        self.memory_bank = torch.empty([0, 256])
        self.mean = 0
        self.std = 1

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
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        return unorganized_pc_no_zeros, nonzero_indices

    def forward(self, organized_pc):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(organized_pc)
        xyz = unorganized_pc_no_zeros.to(self.device).contiguous()
        xyz_feature_maps, center, neighbor_idx, center_idx = self.backbone(xyz)
        interpolated_pc = interpolating_points(xyz, center.permute(0, 2, 1), xyz_feature_maps)

        xyz_patch = torch.cat(xyz_feature_maps, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))

        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        return {
            'xyz_patch': xyz_patch,
            'xyz_patch_full_resized': xyz_patch_full_resized,

        }

    def compute_s_s_map(self, patch_feat, feature_map_dims, mask, label):
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
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

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
    for class_name in cfg.classes:
        model = PointPatchCore(cfg.model.group_size, cfg.model.num_group)
        model.to(args.device)

        # extract features
        memory_list = []
        train_loader = get_data_loader("train", class_name=class_name, img_size=cfg.image_size)
        for sample in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            ret = model(sample['resized_organized_pc'])
            memory_list.append(ret['xyz_patch'])
        memory_bank = torch.cat(memory_list, 0)

        if cfg.do_normalization:
            model.mean = torch.mean(memory_bank)  # save mean std for testing
            model.std = torch.std(memory_bank)
        memory_bank = (memory_bank - model.mean) / model.std

        # coreset sampling
        print(f'\n\nRunning coreset on class {class_name}...')
        model.memory_bank = run_coreset(memory_bank, cfg.coreset.ratio, cfg.coreset.eps)

        # evaluation
        test_loader = get_data_loader("test", class_name=class_name, img_size=cfg.image_size)
        for sample in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
            ret = model(sample['resized_organized_pc'])
            model.compute_s_s_map(ret['xyz_patch'], ret['xyz_patch_full_resized'][0].shape[-2:],
                                  sample['gt_mask'], sample['label'])


def parse_args():
    parser = argparse.ArgumentParser(
        description='')
    parser.add_argument('--config', '-c', help='config file path.', default='reformat/configs/point.py')
    parser.add_argument("--device", default='gpu', type=str, choices=['gpu', 'cpu'])
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

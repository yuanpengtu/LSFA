"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import roc_auc_score

from models.feature_fusion import FeatureFusionBlock
from models.models import Model
from models.pointnet2_utils import interpolating_points
from utils.au_pro_util import calculate_au_pro
from utils.utils import KNNGaussianBlur
from utils.utils import get_coreset_idx_randomp
from utils.utils import set_seeds


class Features(torch.nn.Module):

    def __init__(self, args, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.deep_feature_extractor = Model(
            device=self.device,
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group
        )
        self.deep_feature_extractor.to(self.device)

        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps

        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_xyz_lib = []
        self.patch_rgb_lib = []
        self.patch_fusion_lib = []
        self.patch_lib = []

        self.xyz_dim = 0
        self.rgb_dim = 0

        self.xyz_mean = 0
        self.xyz_std = 0
        self.rgb_mean = 0
        self.rgb_std = 0
        self.fusion_mean = 0
        self.fusion_std = 0

        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.resize2 = torch.nn.AdaptiveAvgPool2d((56, 56))

        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.ins_id = 0

        self.fusion = FeatureFusionBlock(1152, 768, mlp_ratio=4.)

        ckpt = torch.load(args.fusion_module_path)['model']

        incompatible = self.fusion.load_state_dict(ckpt, strict=False)

        print('[Fusion Block]', incompatible)

        self.detect_fuser = linear_model.SGDOneClassSVM(random_state=42)
        self.seg_fuser = linear_model.SGDOneClassSVM(random_state=42)

        self.s_lib = []
        self.s_map_lib = []

    def __call__(self, rgb, xyz):
        # Extract the desired feature maps using the backbone model.
        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        with torch.no_grad():
            rgb_feat, xyz_feat, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)

        interpolate = True
        if interpolate:
            interpolated_feat = interpolating_points(xyz, center.permute(0, 2, 1), xyz_feat).to("cpu")

        xyz_feat = [fmap.to("cpu") for fmap in [xyz_feat]]
        rgb_feat = [fmap.to("cpu") for fmap in [rgb_feat]]

        if interpolate:
            return rgb_feat, xyz_feat, center, ori_idx, center_idx, interpolated_feat
        else:
            return rgb_feat, xyz_feat, center, ori_idx, center_idx

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def add_sample_to_late_fusion_mem_bank(self, sample):
        raise NotImplementedError

    def compute_s_s_map(self, xyz_patch, rgb_patch, fusion_patch, feature_map_dims, mask, label, center, neighbour_idx,
                        nonzero_indices, xyz, center_idx):
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        fusion_patch = (fusion_patch - self.fusion_mean) / self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        s = torch.tensor([[s_xyz, 0.1 * s_rgb, s_fusion]])

        s_map = torch.cat([s_map_xyz, 0.1 * s_map_rgb, s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        s = torch.tensor(self.detect_fuser.score_samples(s))

        s_map = torch.tensor(self.seg_fuser.score_samples(s_map))

        s_map = s_map.view(1, self.image_size, self.image_size)

        self.image_preds.append(s.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(s_map.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())

    def compute_single_s_s_map(self, patch, dist, feature_map_dims, modal='xyz'):

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch

        if modal == 'xyz':
            m_star = self.patch_xyz_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_xyz_lib)  # find knn to m_star pt.1
        elif modal == 'rgb':
            m_star = self.patch_rgb_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_rgb_lib)  # find knn to m_star pt.1
        else:
            m_star = self.patch_fusion_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
            w_dist = torch.cdist(m_star, self.patch_fusion_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # sparse reweight
        # if modal=='rgb':
        #     _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2
        # else:
        #     _, nn_idx = torch.topk(w_dist, k=4*self.n_reweight, largest=False)  # pt.2

        # equation 7 from the paper
        if modal == 'xyz':
            m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1:]], dim=1)
        elif modal == 'rgb':
            m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        else:
            m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1:]], dim=1)

        # if modal=='xyz':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_xyz_lib[nn_idx[0, 1::4]], dim=1) 
        # elif modal=='rgb':
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_rgb_lib[nn_idx[0, 1:]], dim=1)
        # else:
        #     m_star_knn = torch.linalg.norm(m_test - self.patch_fusion_lib[nn_idx[0, 1::4]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))

        s = w * s_star

        # segmentation map
        s_map = min_val.view(1, 1, *feature_map_dims)
        s_map = torch.nn.functional.interpolate(s_map, size=(self.image_size, self.image_size), mode='bilinear')
        s_map = self.blur(s_map)

        return s, s_map

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def save_prediction_maps(self, output_path, rgb_path, save_num=5):
        for i in range(max(save_num, len(self.predictions))):

            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            im = ax.imshow(self.predictions[i], cmap=plt.cm.jet)
            plt.colorbar(im)
            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(self.gts[i], cmap=plt.cm.gray)
            ax3 = fig.add_subplot(1, 3, 3)
            gt = plt.imread(rgb_path[i][0])
            ax3.imshow(gt)

            class_dir = os.path.join(output_path, rgb_path[i][0].split('/')[-5])
            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            ad_dir = os.path.join(class_dir, rgb_path[i][0].split('/')[-3])
            if not os.path.exists(ad_dir):
                os.mkdir(ad_dir)

            plt.savefig(os.path.join(ad_dir, str(self.image_preds[i]) + '_' + rgb_path[i][0].split('/')[-1] + '.jpg'))

    def run_coreset(self):
        self.patch_xyz_lib = torch.cat(self.patch_xyz_lib, 0)
        self.patch_rgb_lib = torch.cat(self.patch_rgb_lib, 0)
        self.patch_fusion_lib = torch.cat(self.patch_fusion_lib, 0)

        self.xyz_mean = torch.mean(self.patch_xyz_lib)
        self.xyz_std = torch.std(self.patch_rgb_lib)
        self.rgb_mean = torch.mean(self.patch_xyz_lib)
        self.rgb_std = torch.std(self.patch_rgb_lib)
        self.fusion_mean = torch.mean(self.patch_xyz_lib)
        self.fusion_std = torch.std(self.patch_rgb_lib)

        self.patch_xyz_lib = (self.patch_xyz_lib - self.xyz_mean) / self.xyz_std
        self.patch_rgb_lib = (self.patch_rgb_lib - self.rgb_mean) / self.rgb_std
        self.patch_fusion_lib = (self.patch_fusion_lib - self.fusion_mean) / self.fusion_std

        if self.f_coreset < 1:
            self.coreset_idx = get_coreset_idx_randomp(self.patch_xyz_lib,
                                                       n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                       eps=self.coreset_eps, )
            self.patch_xyz_lib = self.patch_xyz_lib[self.coreset_idx]
            self.coreset_idx = get_coreset_idx_randomp(self.patch_rgb_lib,
                                                       n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                       eps=self.coreset_eps, )
            self.patch_rgb_lib = self.patch_rgb_lib[self.coreset_idx]
            self.coreset_idx = get_coreset_idx_randomp(self.patch_fusion_lib,
                                                       n=int(self.f_coreset * self.patch_xyz_lib.shape[0]),
                                                       eps=self.coreset_eps, )
            self.patch_fusion_lib = self.patch_fusion_lib[self.coreset_idx]

    def run_late_fusion(self):

        self.s_lib = torch.cat(self.s_lib, 0)
        self.s_map_lib = torch.cat(self.s_map_lib, 0)
        print(self.s_lib.shape)
        self.detect_fuser.fit(self.s_lib)
        self.seg_fuser.fit(self.s_map_lib)

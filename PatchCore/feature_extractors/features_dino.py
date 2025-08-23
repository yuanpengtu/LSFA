"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from models.models import Model
from models.pointnet2_utils import interpolating_points
from utils.au_pro_util import calculate_au_pro
from utils.utils import KNNGaussianBlur
from utils.utils import get_coreset_idx_randomp
from utils.utils import set_seeds

from models.feature_fusion import FeatureFusionBlock

class Features(torch.nn.Module):

    def __init__(self, args, image_size=224, f_coreset=0.1, coreset_eps=0.9):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        allclass = [
            "bagel",
            "cable_gland",
            "carrot",
            "cookie",
            "dowel",
            "foam",
            "peach",
            "potato",
            "rope",
            "tire",
        ]
        self.classname = allclass[args.classname]
        self.deep_feature_extractor = Model(
            device=self.device,
            rgb_backbone_name=args.rgb_backbone_name,
            xyz_backbone_name=args.xyz_backbone_name,
            group_size=args.group_size,
            num_group=args.num_group,
            classname = self.classname
        )
        self.deep_feature_extractor.to(self.device)

        self.fusion_block = FeatureFusionBlock()
        #msg = self.fusion_block.load_state_dict(torch.load('/youtu_fuxi_team1_ceph/yuanpengtu/noise_3D/M3DM_pretrainfeat/savemodel/wangblocks_distill'+self.classname+'_finetuneblock'+'.pkl'), strict=False)
        #print("fusion_block:", msg)
        weightpath = args.weightpath
        msg = self.fusion_block.load_state_dict(torch.load(weightpath), strict=False)
        print("fusion_block true:", msg)

        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = coreset_eps

        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []

        self.xyz_dim = 0
        self.rgb_dim = 0
        self.lib_mean = []
        self.lib_var = []

        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))

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

        self.mean = 0
        self.std = 0

    def __call__(self, rgb, xyz):
        # Extract the desired feature maps using the backbone model.
        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)
        with torch.no_grad():
            
            rgb_feat, xyz_feat, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)

        interpolated_feat = interpolating_points(xyz, center.permute(0, 2, 1), xyz_feat).to("cpu")

        xyz_feat = [fmap.to("cpu") for fmap in [xyz_feat]]

        rgb_feat = [fmap.to("cpu") for fmap in [rgb_feat]]

        return rgb_feat, xyz_feat, center, ori_idx, center_idx, interpolated_feat

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def compute_s_s_map(self, patch, feature_map_dims, mask, label, center, neighbour_idx, nonzero_indices, xyz,
                        center_idx, nonzero_patch_indices=None):  # predict
        '''
        center: point group center position
        neighbour_idx: each group point index
        nonzero_indices: point indices of original point clouds
        xyz: nonzero point clouds
        '''

        if self.args.method_name != 'Point_MAE':
            patch = (patch - self.mean) / self.std
        dist = torch.cdist(patch, self.patch_lib)

        min_val, min_idx = torch.min(dist, dim=1)

        # print(min_val.shape)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = self.patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, self.patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=self.n_reweight, largest=False)  # pt.2

        m_star_knn = torch.linalg.norm(m_test - self.patch_lib[nn_idx[0, 1:]], dim=1)

        D = torch.sqrt(torch.tensor(patch.shape[1]))
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

    def calculate_metrics(self):  # evaluate
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def save_prediction_maps(self, output_path, rgb_path, save_num=5):  # evaluate
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

    def run_coreset(self):  # fit

        self.patch_lib = torch.cat(self.patch_lib, 0).detach().cpu()
        if self.args.method_name != 'Point_MAE':
            self.mean = torch.mean(self.patch_lib)
            self.std = torch.std(self.patch_lib)
            self.patch_lib = (self.patch_lib - self.mean) / self.std

        if self.f_coreset < 1:
            self.coreset_idx = get_coreset_idx_randomp(self.patch_lib,
                                                       n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                       eps=self.coreset_eps, )
            self.patch_lib = self.patch_lib[self.coreset_idx]

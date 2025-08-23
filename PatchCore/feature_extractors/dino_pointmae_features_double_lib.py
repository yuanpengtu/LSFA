import math
import os

import numpy as np

from feature_extractors.features_double_lib import Features
from utils.mvtec3d_util import *


class DoubleRGBPointFeatures(Features):
    def preprocess(self, sample):
        organized_pc = sample
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        return unorganized_pc_no_zeros, nonzero_indices

    def xyz_feat_to_patch(self, xyz_feat, interpolated_pc, nonzero_indices):
        xyz_patch = torch.cat(xyz_feat, 1)
        #xyz_patch = xyz_patch.squeeze(0)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        return xyz_patch, xyz_patch_full_resized

    def forward(self, sample, mask=None, label=None, training=True, class_name=None):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample[1])
        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())
        rgb_patch = torch.cat(rgb_feat, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T


        """
        point_savedict = sample[-3]
        xyz_feat = point_savedict[0]
        center = point_savedict[1].squeeze(0)
        neighbor_idx = point_savedict[2].squeeze(0)
        nonzero_indices = point_savedict[3].squeeze(0)
        unorganized_pc_no_zeros = point_savedict[4].squeeze(0)
        center_idx = point_savedict[5].squeeze(0)
        rgb_patch = point_savedict[6].squeeze(0)
        interpolated_pc = point_savedict[7].squeeze(0)
        """
        xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)


        self.fusion_block= self.fusion_block.cuda()
        self.fusion_block.eval()
        rgb_patch = rgb_patch.reshape(1, 3136, 768)
        xyz_patch = xyz_patch.reshape(1, 3136, 1152)
        with torch.no_grad():
            xyz_patch, rgb_patch = self.fusion_block.feature_fusion(xyz_patch.cuda(), rgb_patch.cuda())
        rgb_patch = rgb_patch.squeeze(0).cpu()
        xyz_patch = xyz_patch.squeeze(0).cpu()

        if training:
            rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784 * 4, -1)
            patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)
            if class_name is not None:
                torch.save(patch, os.path.join('datasets/patch_lib', class_name + str(self.ins_id) + '.pt'))
                self.ins_id += 1
            self.patch_xyz_lib.append(xyz_patch)
            self.patch_rgb_lib.append(rgb_patch)
        else:
            self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center,
                                 neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.forward(sample, class_name=class_name)

    def predict(self, sample, mask, label):
        self.forward(sample, mask, label, training=False)

    def add_sample_to_late_fusion_mem_bank(self, sample):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample[1])
        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feat, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)
        """
        point_savedict = sample[-3]
        xyz_feat = point_savedict[0]
        center = point_savedict[1].squeeze(0)
        neighbor_idx = point_savedict[2].squeeze(0)
        nonzero_indices = point_savedict[3].squeeze(0)
        unorganized_pc_no_zeros = point_savedict[4].squeeze(0)
        center_idx = point_savedict[5].squeeze(0)
        rgb_patch = point_savedict[6].squeeze(0)
        interpolated_pc = point_savedict[7].squeeze(0)

        xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)
        """

        #if self.classname == 'cable_gland' or self.classname == 'foam' or self.classname == 'bagel':
        self.fusion_block= self.fusion_block.cuda()
        self.fusion_block.eval()
        rgb_patch = rgb_patch.reshape(1, 3136, 768)
        xyz_patch = xyz_patch.reshape(1, 3136, 1152)
        rgb_patch_ori, xyz_patch_ori = rgb_patch, xyz_patch
        with torch.no_grad():
            xyz_patch, rgb_patch = self.fusion_block.feature_fusion(xyz_patch.cuda(), rgb_patch.cuda())
        rgb_patch = rgb_patch.squeeze(0).cpu()
        xyz_patch = xyz_patch.squeeze(0).cpu()

        # 2D dist 
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))



        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')

        s = torch.tensor([[s_xyz, 0.1 * s_rgb]])
        s_map = torch.cat([s_map_xyz, 0.1 * s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)



        #2D dist2
        
        rgb_patch_ori = rgb_patch_ori.squeeze(0).cpu()
        xyz_patch_ori = xyz_patch_ori.squeeze(0).cpu()
        xyz_patch_ori = (xyz_patch_ori - self.xyz_mean) / self.xyz_std
        rgb_patch_ori = (rgb_patch_ori - self.rgb_mean) / self.rgb_std
        dist_xyz_ori = torch.cdist(xyz_patch_ori, self.patch_xyz_lib)
        dist_rgb_ori = torch.cdist(rgb_patch_ori, self.patch_rgb_lib)

        rgb_feat_size_ori = (int(math.sqrt(rgb_patch_ori.shape[0])), int(math.sqrt(rgb_patch_ori.shape[0])))
        xyz_feat_size_ori = (int(math.sqrt(xyz_patch_ori.shape[0])), int(math.sqrt(xyz_patch_ori.shape[0])))



        s_xyz_ori, s_map_xyz_ori = self.compute_single_s_s_map(xyz_patch_ori, dist_xyz_ori, xyz_feat_size_ori, modal='xyz')
        s_rgb_ori, s_map_rgb_ori = self.compute_single_s_s_map(rgb_patch_ori, dist_rgb_ori, rgb_feat_size_ori, modal='rgb')

        s_ori = torch.tensor([[s_xyz_ori, 0.1 * s_rgb_ori]])
        s_map_ori = torch.cat([s_map_xyz_ori, 0.1 * s_map_rgb_ori], dim=0).squeeze().reshape(2, -1).permute(1, 0)



        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

        self.s_lib_ori.append(s_ori)
        self.s_map_lib_ori.append(s_map_ori)

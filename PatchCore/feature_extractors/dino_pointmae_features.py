import os

import numpy as np

from feature_extractors.features_dino import Features
from utils.mvtec3d_util import *


class RGBPointFeatures(Features):
    def preprocess(self, sample):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        return unorganized_pc_no_zeros, nonzero_indices

    def xyz_feat_to_patch(self, xyz_feat, interpolated_pc, nonzero_indices):
        xyz_patch = torch.cat(xyz_feat, 1)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        return xyz_patch, xyz_patch_full_resized

    def forward(self, sample, mask=None, label=None, training=True, class_name=None):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample)
        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feat, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)

        rgb_patch = rgb_patch.reshape(1, 3136, 768)
        xyz_patch = xyz_patch.reshape(1, 3136, 1152)
        self.fusion_block = self.fusion_block.cuda()
        self.fusion_block.eval()
        with torch.no_grad():
            fuse_feat = self.fusion_block.feature_fusion(xyz_patch.cuda(), rgb_patch.cuda())
            
        fuse_feat = fuse_feat.view(56*56, -1).cpu()
        #print(fuse_feat.shape)

        patch = fuse_feat#torch.cat([xyz_patch, rgb_patch], dim=1)
        


        if training:
            if class_name is not None:
                torch.save(patch, os.path.join('datasets/patch_lib', class_name + str(self.ins_id) + '.pt'))
                self.ins_id += 1

            self.patch_lib.append(patch)
        else:
            self.compute_s_s_map(patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center, neighbor_idx,
                                 nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.forward(sample, class_name=class_name)

    def predict(self, sample, mask, label):
        self.forward(sample, mask, label, training=False)

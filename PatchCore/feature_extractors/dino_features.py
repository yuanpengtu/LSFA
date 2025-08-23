import os

import numpy as np

from feature_extractors.features_dino import Features
from utils.mvtec3d_util import *


class RGBFeatures(Features):
    def preprocess(self, sample):
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        unorganized_pc_no_zeros = torch.tensor(unorganized_pc[nonzero_indices, :]).unsqueeze(dim=0).permute(0, 2, 1)
        return unorganized_pc_no_zeros, nonzero_indices


    def forward(self, sample, mask=None, label=None, training=True, class_name=None):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample)


        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, _ = self(sample[0],
                                                                       unorganized_pc_no_zeros.contiguous())
        
        
        rgb_patch = torch.cat(rgb_feat, 1)
        xyz_patch = torch.cat(xyz_feat, 1)
        
        self.fusion_block= self.fusion_block.cuda()
        self.fusion_block.eval()
        rgb_patch = rgb_patch.reshape(1, 3136, 768)
        xyz_patch = xyz_patch.reshape(1, 1024, 1152)
        with torch.no_grad():
            rgb_patch = self.fusion_block.rgb_mlp(rgb_patch.cuda())
        rgb_patch = rgb_patch.squeeze(0).cpu()
        xyz_patch = xyz_patch.squeeze(0).cpu()
        #rgb_patch = torch.load(sample[-1][0])
        #rgb_patch = rgb_patch.reshape(56*56, rgb_patch.shape[1])
        #rgb_patch = rgb_patch.reshape(1, 1152, 56, 56).cpu()
        #rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        patch = rgb_patch
        if training:
            if class_name is not None:
                torch.save(patch, os.path.join('datasets/patch_lib', class_name + str(self.ins_id) + '.pt'))
                self.ins_id += 1

            self.patch_lib.append(patch)
        else:
            self.compute_s_s_map(patch, rgb_feat[0].shape[-2:], mask, label, center, neighbor_idx,
                                 nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.forward(sample, class_name=class_name)

    def predict(self, sample, mask, label):
        self.forward(sample, mask, label, training=False)

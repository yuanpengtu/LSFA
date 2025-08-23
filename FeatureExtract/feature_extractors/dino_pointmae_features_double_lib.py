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
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        return xyz_patch, xyz_patch_full_resized

    def forward(self, sample, mask=None, label=None, training=True, class_name=None):
        """
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample[1])
        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feat, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)
        """
        
        #self.fusion_block= self.fusion_block.cuda()
        #self.fusion_block.eval()
        #rgb_patch = rgb_patch.reshape(1, 3136, 768)
        #xyz_patch = xyz_patch.reshape(1, 3136, 1152)
        #with torch.no_grad():
        #    xyz_patch, rgb_patch = self.fusion_block.feature_fusion(xyz_patch.cuda(), rgb_patch.cuda())
        #rgb_patch = rgb_patch.squeeze(0).cpu()
        #xyz_patch = xyz_patch.squeeze(0).cpu()

        if True:
            #rgb_patch_resize = rgb_patch.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784 * 4, -1)
            #patch = torch.cat([xyz_patch, rgb_patch_resize], dim=1)
            #if class_name is not None:
            #    torch.save(patch, os.path.join('datasets/patch_lib', class_name + str(self.ins_id) + '.pt'))
            #    self.ins_id += 1
            #self.patch_xyz_lib.append(xyz_patch)
            #self.patch_rgb_lib.append(rgb_patch)

            #rgb_patch_rotate90 = sample[3]
            xyz_patch = sample[1]
            #xyz_patch_rotate90 = sample[4]
            #xyz_patch_rotate180 = sample[5]
            #xyz_patch_rotate270 = sample[6]
            #unorganized_pc_no_zeros_rotate90, nonzero_indices_rotate90 = self.preprocess(xyz_patch_rotate90)
            #unorganized_pc_no_zeros_rotate180, nonzero_indices_rotate180 = self.preprocess(xyz_patch_rotate180)
            #unorganized_pc_no_zeros_rotate270, nonzero_indices_rotate270 = self.preprocess(xyz_patch_rotate270)
            unorganized_pc_no_zeros, nonzero_indices = self.preprocess(xyz_patch)

            #xyz_patch_fliph = sample[7]
            #xyz_patch_flipv = sample[8]
            #unorganized_pc_no_zeros_fliph, nonzero_indices_fliph = self.preprocess(xyz_patch_fliph)
            #unorganized_pc_no_zeros_flipv, nonzero_indices_flipv = self.preprocess(xyz_patch_flipv)

            #rgb_feat_fliph, xyz_feat_fliph, center_fliph, neighbor_idx_fliph, center_idx_fliph, interpolated_pc_fliph = self(sample[3], unorganized_pc_no_zeros_fliph.contiguous())
            #rgb_patch_fliph = torch.cat(rgb_feat_fliph, 1)
            #rgb_patch_fliph = rgb_patch_fliph.reshape(rgb_patch_fliph.shape[1], -1).T
            #xyz_patch_fliph, xyz_patch_full_resized_fliph = self.xyz_feat_to_patch(xyz_feat_fliph, interpolated_pc_fliph, nonzero_indices_fliph)
            rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0], unorganized_pc_no_zeros.contiguous())
            rgb_patch = torch.cat(rgb_feat, 1)
            rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
            xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)            





            region_rgb1, region_rgb2, region_xyz1, region_xyz2 = sample[-3]
            region_unorganized_pc_no_zeros1, region_nonzero_indices1 = self.preprocess(region_xyz1)
            region_unorganized_pc_no_zeros2, region_nonzero_indices2 = self.preprocess(region_xyz2)

            region_rgb_feat1, region_xyz_feat1, _, _, _, region_interpolated_pc1 = self(region_rgb1, region_unorganized_pc_no_zeros1.contiguous())
            region_rgb_patch1 = torch.cat(region_rgb_feat1, 1)
            region_rgb_patch1 = region_rgb_patch1.reshape(region_rgb_patch1.shape[1], -1).T
            region_xyz_patch1, _ = self.xyz_feat_to_patch(region_xyz_feat1, region_interpolated_pc1, region_nonzero_indices1)   


            region_rgb_feat2, region_xyz_feat2, _, _, _, region_interpolated_pc2 = self(region_rgb2, region_unorganized_pc_no_zeros2.contiguous())
            region_rgb_patch2 = torch.cat(region_rgb_feat2, 1)
            region_rgb_patch2 = region_rgb_patch2.reshape(region_rgb_patch2.shape[1], -1).T
            region_xyz_patch2, _ = self.xyz_feat_to_patch(region_xyz_feat2, region_interpolated_pc2, region_nonzero_indices2)   
            #rgb_feat_flipv, xyz_feat_flipv, center_flipv, neighbor_idx_flipv, center_idx_flipv, interpolated_pc_flipv = self(sample[3], unorganized_pc_no_zeros_flipv.contiguous())
            #rgb_patch_flipv = torch.cat(rgb_feat_flipv, 1)
            #rgb_patch_flipv = rgb_patch_flipv.reshape(rgb_patch_flipv.shape[1], -1).T
            #xyz_patch_flipv, xyz_patch_full_resized_flipv = self.xyz_feat_to_patch(xyz_feat_flipv, interpolated_pc_flipv, nonzero_indices_flipv)
            


            #rgb_feat_rotate90, xyz_feat_rotate90, center_rotate90, neighbor_idx_rotate90, center_idx_rotate90, interpolated_pc_rotate90 = self(sample[3], unorganized_pc_no_zeros_rotate90.contiguous())
            #rgb_patch_rotate90 = torch.cat(rgb_feat_rotate90, 1)
            #rgb_patch_rotate90 = rgb_patch_rotate90.reshape(rgb_patch_rotate90.shape[1], -1).T
            #xyz_patch_rotate90, xyz_patch_full_resized_rotate90 = self.xyz_feat_to_patch(xyz_feat_rotate90, interpolated_pc_rotate90, nonzero_indices_rotate90)
            
            #rgb_feat_rotate180, xyz_feat_rotate180, center_rotate180, neighbor_idx_rotate180, center_idx_rotate180, interpolated_pc_rotate180 = self(sample[3], unorganized_pc_no_zeros_rotate180.contiguous())
            #rgb_patch_rotate180 = torch.cat(rgb_feat_rotate180, 1)
            #rgb_patch_rotate180 = rgb_patch_rotate180.reshape(rgb_patch_rotate180.shape[1], -1).T
            #xyz_patch_rotate180, xyz_patch_full_resized_rotate180 = self.xyz_feat_to_patch(xyz_feat_rotate180, interpolated_pc_rotate180, nonzero_indices_rotate180)
            
            #rgb_feat_rotate270, xyz_feat_rotate270, center_rotate270, neighbor_idx_rotate270, center_idx_rotate270, interpolated_pc_rotate270 = self(sample[3], unorganized_pc_no_zeros_rotate270.contiguous())
            #rgb_patch_rotate270 = torch.cat(rgb_feat_rotate270, 1)
            #rgb_patch_rotate270 = rgb_patch_rotate270.reshape(rgb_patch_rotate270.shape[1], -1).T
            #xyz_patch_rotate270, xyz_patch_full_resized_rotate270 = self.xyz_feat_to_patch(xyz_feat_rotate270, interpolated_pc_rotate270, nonzero_indices_rotate270)
            

            #rgb_patch_resize_rotate90 = rgb_patch_rotate90.repeat(4, 1).reshape(784, 4, -1).permute(1, 0, 2).reshape(784 * 4, -1)
            #patch_rotate90 = torch.cat([xyz_patch_rotate90, rgb_patch_resize_rotate90], dim=1)
            #if class_name is not None:
            #    torch.save(patch, os.path.join('datasets/patch_lib', class_name + str(self.ins_id) + '.pt'))
            #    self.ins_id += 1
            #self.patch_xyz_lib.append(xyz_patch_rotate90)
            #self.patch_rgb_lib.append(rgb_patch_rotate90)
            #path_img = sample[-2]
            #path_point = sample[-1]
            #path_point = path_point[0].replace('.', 'full.')
            #torch.save(xyz_patch, path_point)
            #path_point_fliph = path_point[0].replace('.', 'fliph.')
            #path_point_flipv = path_point[0].replace('.', 'flipv.')
            #torch.save(xyz_patch_fliph, path_point_fliph)
            #torch.save(xyz_patch_flipv, path_point_flipv)
            #print(path_point_fliph, path_point_flipv)

            #path_point_90 = path_point[0].replace('.', '90.')
            #path_point_180 = path_point[0].replace('.', '180.')
            #path_point_270 = path_point[0].replace('.', '270.')
            #torch.save(xyz_patch, path_img[0])
            #print(xyz_patch_rotate90.shape)
            #torch.save(xyz_patch_rotate90, path_point_90)
            #torch.save(xyz_patch_rotate180, path_point_180)
            #torch.save(xyz_patch_rotate270, path_point_270)
            #print(path_point)


        #else:
            #self.compute_s_s_map(xyz_patch, rgb_patch, xyz_patch_full_resized[0].shape[-2:], mask, label, center,
            #                     neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(), center_idx)
        point_savedict = {}
        path_img = sample[-2]
        path_point = sample[-1]
        path_point = path_point[0].replace('.', 'full.')
        path_point = path_point.replace('.pt', '.pth')
        point_savedict['xyz_feat'] = xyz_feat
        point_savedict['center'] = center
        point_savedict['neighbor_idx'] = neighbor_idx
        point_savedict['nonzero_indices'] = nonzero_indices
        point_savedict['unorganized_pc_no_zeros'] = unorganized_pc_no_zeros
        point_savedict['center_idx'] = center_idx
        point_savedict['rgb_patch'] = rgb_patch.detach().cpu()
        point_savedict['interpolated_pc'] = interpolated_pc




        point_savedict['region1_xyz_feat'] = region_xyz_patch1
        point_savedict['region2_xyz_feat'] = region_xyz_patch2
        point_savedict['region1_rgb_patch'] = region_rgb_patch1.detach().cpu()
        point_savedict['region2_rgb_patch'] = region_rgb_patch2.detach().cpu()


        
        print(region_rgb_patch1.shape)

        torch.save(point_savedict, path_point)
        #torch.save(xyz_patch, path_img[0])
        #torch.save(rgb_patch, path_point[0])
        print(path_point)

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

        #self.fusion_block= self.fusion_block.cuda()
        #self.fusion_block.eval()
        #rgb_patch = rgb_patch.reshape(1, 3136, 768)
        #xyz_patch = xyz_patch.reshape(1, 3136, 1152)
        #with torch.no_grad():
        #    xyz_patch, rgb_patch = self.fusion_block.feature_fusion(xyz_patch.cuda(), rgb_patch.cuda())
        #rgb_patch = rgb_patch.squeeze(0).cpu()
        #xyz_patch = xyz_patch.squeeze(0).cpu()


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

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

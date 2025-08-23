import math
import os

import numpy as np

from feature_extractors.features_triple_lib import Features
from utils.mvtec3d_util import *

fusion_block = True


class TripleFeatures(Features):
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
        return xyz_patch, xyz_patch_full_resized, xyz_patch_full_2d

    def forward(self, sample, mask=None, label=None, training=True, class_name=None):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample)
        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feat, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        xyz_patch, xyz_patch_full_resized, xyz_patch_full_2d = self.xyz_feat_to_patch(xyz_feat, interpolated_pc,
                                                                                      nonzero_indices)
        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if fusion_block:
            with torch.no_grad():
                patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            patch = patch.reshape(-1, patch.shape[2]).detach()
        else:
            patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)

        if training:
            if class_name is not None:
                torch.save(patch, os.path.join('datasets/patch_lib_tri', class_name + str(self.ins_id) + '.pt'))
                self.ins_id += 1

            self.patch_xyz_lib.append(xyz_patch)
            self.patch_rgb_lib.append(rgb_patch)
            self.patch_fusion_lib.append(patch)
        else:
            self.compute_s_s_map(xyz_patch, rgb_patch, patch, xyz_patch_full_resized[0].shape[-2:], mask, label,
                                 center, neighbor_idx, nonzero_indices, unorganized_pc_no_zeros.contiguous(),
                                 center_idx)

    def add_sample_to_mem_bank(self, sample, class_name=None):
        self.forward(sample, class_name=class_name)

    def predict(self, sample, mask, label):
        self.forward(sample, mask, label, training=False)

    def add_sample_to_late_fusion_mem_bank(self, sample):
        unorganized_pc_no_zeros, nonzero_indices = self.preprocess(sample)
        rgb_feat, xyz_feat, center, neighbor_idx, center_idx, interpolated_pc = self(sample[0],
                                                                                     unorganized_pc_no_zeros.contiguous())

        rgb_patch = torch.cat(rgb_feat, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T

        xyz_patch, xyz_patch_full_resized, xyz_patch_full_2d = self.xyz_feat_to_patch(xyz_feat, interpolated_pc,
                                                                                      nonzero_indices)
        xyz_patch_full_resized2 = self.resize2(self.average(xyz_patch_full_2d))
        xyz_patch2 = xyz_patch_full_resized2.reshape(xyz_patch_full_resized2.shape[1], -1).T

        rgb_patch_size = int(math.sqrt(rgb_patch.shape[0]))
        rgb_patch2 = self.resize2(rgb_patch.permute(1, 0).reshape(-1, rgb_patch_size, rgb_patch_size))
        rgb_patch2 = rgb_patch2.reshape(rgb_patch.shape[1], -1).T

        if fusion_block:
            with torch.no_grad():
                patch = self.fusion.feature_fusion(xyz_patch2.unsqueeze(0), rgb_patch2.unsqueeze(0))
            patch = patch.reshape(-1, patch.shape[2]).detach()
        else:
            patch = torch.cat([xyz_patch2, rgb_patch2], dim=1)
#
        # 工作内容/项目名称	角色/工作职责	个人贡献与产出	专业能力体现   项目相关人
        # 宁德时代极耳质检项目   模块负责人
        # 多景深图像合成

        # 苹果无线充电模组P193外观质检项目(昆山立讯/常州信维)  子项目负责人
        # 个人贡献与产出
        # 负责与常州信维和苹果方面的对接和驻场维保，负责项目中各型号的配准模块、3项Top级缺陷、4项新缺陷的研发。
        # 保障算法以技术优势在与竞品阿丘科技的PK中胜出（EVT准确率：优图98% vs 阿丘90%，腾讯获得全部机台订单，阿丘出局），协助业务侧预计获得约800万元订单。
        # 支撑算法从样机到量产的全流程研发和问题响应，完成算法在全部常州/昆山两地共计24条产线机台的量产交付，并促成启动下一代P195型号的订单签单。
        # 打造用户口碑，作为标杆项目对苹果/立讯/信维客户进行团队技术能力介绍，促成团队首次获得浙江立讯-iwatch订单，并已获推荐江西立讯-airpods项目。
        # 注重总结沉淀,
        # 专业能力体现

        # 工作内容/项目名称	角色/工作职责	个人贡献与产出	专业能力体现   项目相关人
        # 苹果无线充电模组P193外观质检项目(昆山立讯/常州信维)  总负责人
        # - 个人贡献与产出
        # 负责与常州信维和苹果方面的对接和驻场维保，负责项目中各型号的配准模块、3项Top级缺陷、4项新缺陷的研发。
        # 注重提升人效，人力投入相比上一代项目降低 80% 以上，
        # - 专业能力体现

        # 工业 AI 影响力建设
        # 子项目负责人

        # I3 引擎与质检工具研发
        # 重要参与者

        # 写文档，
        # I3 引擎 - 目标检测选型对比文档 https://iwiki.woa.com/pages/viewpage.action?pageId=1971617289
        # I3 引擎 - 目标检测方法沉淀 https://iwiki.woa.com/pages/viewpage.action?pageId=1985139094
        # 无线充电模组项目技术方案总结
        # 模组类项目算法设计原则沉淀
        # 培养人，代码仓库。

        # 3D dist 
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        fusion_patch = (patch - self.fusion_mean) / self.fusion_std

        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        dist_fusion = torch.cdist(fusion_patch, self.patch_fusion_lib)

        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        fusion_feat_size = (int(math.sqrt(fusion_patch.shape[0])), int(math.sqrt(fusion_patch.shape[0])))

        # 3 memory bank results
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s_fusion, s_map_fusion = self.compute_single_s_s_map(fusion_patch, dist_fusion, fusion_feat_size,
                                                             modal='fusion')

        # input s and s_map  
        s = torch.tensor([[s_xyz, 0.1 * s_rgb, s_fusion]])
        s_map = torch.cat([s_map_xyz, 0.1 * s_map_rgb, s_map_fusion], dim=0).squeeze().reshape(3, -1).permute(1, 0)

        self.s_lib.append(s)
        self.s_map_lib.append(s_map)



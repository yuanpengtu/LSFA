import os
from PIL import Image
from torchvision import transforms
import glob
from torch.utils.data import Dataset
from utils.mvtec3d_util import *
from torch.utils.data import DataLoader
import numpy as np
import random
import torch
DATASETS_PATH = './dataset/mvtec3d_preprocessed'
RGB_SIZE = 224





def mvtec3d_classes():
    return [
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
from randaugment import RandAugmentMC

class MVTec3D(Dataset):

    def __init__(self, split, class_name, img_size):
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.cls = class_name
        self.size = img_size
        self.img_path = os.path.join(DATASETS_PATH, self.cls, split)
        self.rgb_transform = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])
         
        self.rgb_transform_v2 = transforms.Compose(
            [transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
             transforms.RandomHorizontalFlip(),
             transforms.RandomHorizontalFlip(),
             RandAugmentMC(n=1, m=3),
             transforms.ToTensor(),
             transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)])

class MVTec3DPreTrainTensor(Dataset):
    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path
        self.tensor_paths = os.listdir(self.root_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.root_path, tensor_path))

        label = 0

        return tensor, label

class MVTec3DTrain(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="train", class_name=class_name, img_size=img_size)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)
        return (img, resized_organized_pc, resized_depth_map_3channel), label




from collections import deque
import numpy as np
import math
class MVTec3DTrainALL(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="train", class_name=class_name, img_size=img_size)
        self.img_paths, self.labels, self.classlabels, self.img_tot_paths_paste = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.allclasses = [
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
        self.patch_queue = deque(maxlen=15) 
        #self.group_divider = Group(num_group=1024, group_size=128)
    def load_dataset(self):
        allclasses = [
            #"bagel",
            #"cable_gland",
            #"carrot",
            #"cookie",
            #"dowel",
            "foam",
            #"peach",
            #"potato",
            #"rope",
            #"tire",
        ]
        img_tot_paths = []
        tot_labels = []
        lens = 0
        classlabels = []
        count = 0
        for cls in allclasses:
            rgb_paths = glob.glob(os.path.join(DATASETS_PATH, cls, 'train', 'good', 'rgb') + "/*.png")
            tiff_paths = glob.glob(os.path.join(DATASETS_PATH, cls, 'train', 'good', 'xyz') + "/*.tiff")
            rgb_paths.sort()
            tiff_paths.sort()
            sample_paths = list(zip(rgb_paths, tiff_paths))
            img_tot_paths.extend(sample_paths)
            lens += len(sample_paths)
            
            classlabels.extend([count]*len(img_tot_paths))

            count+=1
        tot_labels.extend([0] * lens)

        img_tot_paths_paste = []
        for cls in ["peach", "foam"]:
            rgb_paths_paste = glob.glob(os.path.join(DATASETS_PATH, cls, 'train', 'good', 'rgb') + "/*.png")
            tiff_paths_paste = glob.glob(os.path.join(DATASETS_PATH, cls, 'train', 'good', 'xyz') + "/*.tiff")
            rgb_paths_paste.sort()
            tiff_paths_paste.sort()
            sample_paths_paste = list(zip(rgb_paths_paste, tiff_paths_paste))
            img_tot_paths_paste.extend(sample_paths_paste)


        return img_tot_paths, tot_labels, classlabels, img_tot_paths_paste

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        imgnew = Image.open(rgb_path).convert('RGB')

        img = self.rgb_transform(imgnew)

        #img2 = self.rgb_transform_v2(imgnew)
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)

        
        label_segment = torch.zeros(img.shape)[0, :, :]
        prob_v = np.random.uniform()
        noise_or_not = 0  
        return (img, resized_organized_pc,  resized_depth_map_3channel, self.classlabels[idx], label_segment, noise_or_not), label

class MVTec3DTest(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="test", class_name=class_name, img_size=img_size)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, resized_depth_map_3channel), gt[:1], label, rgb_path


def get_data_loader(split, class_name, img_size):
    if split in ['train']:
        dataset = MVTec3DTrain(class_name=class_name, img_size=img_size)
    elif split in ['test']:
        dataset = MVTec3DTest(class_name=class_name, img_size=img_size)

    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False,
                             pin_memory=True)
    return data_loader





import torch.nn.functional as F
import torch
import torch.nn as nn
class MVTec3DTrainFeaturesALL(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="train", class_name=class_name, img_size=img_size)
        self.img_paths, self.labels, self.classlabels = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.resize = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.image_size = 224
    def load_dataset(self):
        allclasses = [
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
        img_tot_paths = []
        tot_labels = []
        lens = 0
        classlabels = []
        count = 0
        for cls in allclasses:
            rgb_paths = glob.glob(os.path.join(DATASETS_PATH, cls, 'train', 'good', 'rgb') + "/*.png")
            tiff_paths = glob.glob(os.path.join(DATASETS_PATH, cls, 'train', 'good', 'xyz') + "/*.tiff")
            rgb_paths.sort()
            tiff_paths.sort()
            sample_paths = list(zip(rgb_paths, tiff_paths))
            img_tot_paths.extend(sample_paths)
            lens += len(sample_paths)
            classlabels.extend([count]*len(sample_paths))
            count+=1
        tot_labels.extend([0] * lens)
        return img_tot_paths, tot_labels, classlabels

    def __len__(self):
        return len(self.img_paths)
    def xyz_feat_to_patch(self, xyz_feat, interpolated_pc, nonzero_indices):
        xyz_patch = torch.cat(xyz_feat, 1)
        xyz_patch = xyz_patch.squeeze(0)
        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size),
                                     dtype=xyz_patch.dtype)
        xyz_patch_full[:, :, nonzero_indices] = interpolated_pc

        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T
        return xyz_patch, xyz_patch_full_resized

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]


        path_img = rgb_path.replace('.png', 'rgb.pt')
        path_point = tiff_path.replace('.tiff', 'point.pt')
        path_point = path_point.replace('.', 'full.')
        path_point = path_point.replace('.pt', '.pth')
        point_savedict = torch.load(path_point,map_location ='cpu')
        xyz_feat = point_savedict['xyz_feat']
        center = point_savedict['center']
        neighbor_idx = point_savedict['neighbor_idx']
        nonzero_indices = point_savedict['nonzero_indices']
        unorganized_pc_no_zeros = point_savedict['unorganized_pc_no_zeros']
        center_idx = point_savedict['center_idx']
        rgb_patch = point_savedict['rgb_patch']#[1:]
        interpolated_pc = point_savedict['interpolated_pc']

        xyz_patch, xyz_patch_full_resized = self.xyz_feat_to_patch(xyz_feat, interpolated_pc, nonzero_indices)

        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        classlabel = self.classlabels[idx]

        img_rotate_patch = img.clone()
        size_img = img_rotate_patch.shape[-1]
        for i in range(size_img//4):
            for j in range(size_img//4):
                rotnum = random.randint(0,3)
                for _ in range(rotnum):
                    img_rotate_patch[:, 4*i:4*(i+1), 4*j:4*(j+1)] = torch.rot90(img_rotate_patch[:, 4*i:4*(i+1), 4*j:4*(j+1)], 1, [1, 2])

        return (img, classlabel, xyz_patch, rgb_patch, (xyz_patch, xyz_patch, xyz_patch, xyz_patch, img_rotate_patch)), label




class MVTec3DTrainFeatures(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="train", class_name=class_name, img_size=img_size)
        self.img_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        tot_labels = []
        rgb_paths = glob.glob(os.path.join(self.img_path, 'good', 'rgb') + "/*.png")
        tiff_paths = glob.glob(os.path.join(self.img_path, 'good', 'xyz') + "/*.tiff")
        rgb_paths.sort()
        tiff_paths.sort()
        sample_paths = list(zip(rgb_paths, tiff_paths))
        img_tot_paths.extend(sample_paths)
        tot_labels.extend([0] * len(sample_paths))

        return img_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label = self.img_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]
        img_feat = torch.load(rgb_path.replace('.png', 'rgb.pt')).cpu()
        point_feat = torch.load(tiff_path.replace('.tiff', 'point.pt')).cpu()
        img_feat = img_feat.reshape(56*56, img_feat.shape[1])
        img = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img)
        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)




        return (img, resized_organized_pc, resized_depth_map_3channel,img_feat,point_feat), label


class MVTec3DTestFeatures(MVTec3D):
    def __init__(self, class_name, img_size):
        super().__init__(split="test", class_name=class_name, img_size=img_size)
        self.gt_transform = transforms.Compose([
            transforms.Resize((RGB_SIZE, RGB_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])
        self.img_paths, self.gt_paths, self.labels = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                rgb_paths.sort()
                tiff_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))
                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend([0] * len(sample_paths))
                tot_labels.extend([0] * len(sample_paths))
            else:
                rgb_paths = glob.glob(os.path.join(self.img_path, defect_type, 'rgb') + "/*.png")
                tiff_paths = glob.glob(os.path.join(self.img_path, defect_type, 'xyz') + "/*.tiff")
                gt_paths = glob.glob(os.path.join(self.img_path, defect_type, 'gt') + "/*.png")
                rgb_paths.sort()
                tiff_paths.sort()
                gt_paths.sort()
                sample_paths = list(zip(rgb_paths, tiff_paths))

                img_tot_paths.extend(sample_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(sample_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label = self.img_paths[idx], self.gt_paths[idx], self.labels[idx]
        rgb_path = img_path[0]
        tiff_path = img_path[1]

        rgb_feats = img_path[0].replace('.png', 'rgb.pt')
        point_feats = img_path[1].replace('.tiff', 'point.pt')
        rgb_feats = torch.load(rgb_feats)
        point_feats = torch.load(point_feats)
        img_original = Image.open(rgb_path).convert('RGB')
        img = self.rgb_transform(img_original)

        organized_pc = read_tiff_organized_pc(tiff_path)
        depth_map_3channel = np.repeat(organized_pc_to_depth_map(organized_pc)[:, :, np.newaxis], 3, axis=2)
        resized_depth_map_3channel = resize_organized_pc(depth_map_3channel)
        resized_organized_pc = resize_organized_pc(organized_pc)

        if gt == 0:
            gt = torch.zeros(
                [1, resized_depth_map_3channel.size()[-2], resized_depth_map_3channel.size()[-2]])
        else:
            gt = Image.open(gt).convert('L')
            gt = self.gt_transform(gt)
            gt = torch.where(gt > 0.5, 1., .0)

        return (img, resized_organized_pc, resized_depth_map_3channel, rgb_feats, point_feats), gt[:1], label, rgb_path

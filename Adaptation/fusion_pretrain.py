import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
#torch.autograd.set_detect_anomaly(True)
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm

import timm.optim.optim_factory as optim_factory

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler


from engine_fusion_pretrain import train_one_epoch

import data.mvtec3d

import torch
from models.feature_fusion import FeatureFusionBlock
from NTXentLoss import NTXentLoss
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
from models.models import PointTransformer, Group, Block
from collections import deque
def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12, max=50).sqrt() # for numerical stability
    return dist
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1).float()
        x = self.head(x)
        return x.permute(0, 2, 1).float()

class FeatureFusionBlock(nn.Module):
    def __init__(self, xyz_dim=1152, rgb_dim=768, mlp_ratio=4.):
        super().__init__()

        self.xyz_dim = xyz_dim
        self.rgb_dim = rgb_dim
        self.xyz_mlp = nn.Sequential(Block(1152, 12))
        self.rgb_mlp = nn.Sequential(Block(768, 12))
        self.rgb_head = nn.Linear(rgb_dim, 512)
        self.xyz_head = nn.Linear(xyz_dim, 512)

        self.pool_multiscale = torch.nn.AdaptiveAvgPool2d((28, 28))
        self.pool_multiscale2 = torch.nn.AdaptiveAvgPool2d((36, 36))
        self.T = 1

        self.queue_rgb = [[], [], [], [], [], [], [], [], [], []]
        self.queue_point = [[], [], [], [], [], [], [], [], [], []]

        self.queue_rgb_global = [[], [], [], [], [], [], [], [], [], []]
        self.queue_point_global = [[], [], [], [], [], [], [], [], [], []]

        self.bn1 = nn.BatchNorm1d(768, affine=False)
        self.bn2 = nn.BatchNorm1d(1152, affine=False)
        self.start_iteration = 60
        self.banksize = 60000

        self.criterion = NTXentLoss(temperature = 0.1).cuda()


    def feature_fusion(self, xyz_feature, rgb_feature):

        xyz_feature = self.xyz_mlp(xyz_feature)
        rgb_feature = self.rgb_mlp(rgb_feature)
        return xyz_feature, rgb_feature

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * 0).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, xyz_feature, region_xyz_patch1, region_xyz_patch2, rgb_feature, region_rgb_patch1, region_rgb_patch2, img_feats_rotate, imgs, idx, classlabels):
        bs = xyz_feature.shape[0]
        feature_rgb_rotate = img_feats_rotate
        feature_rgb_rotate = self.rgb_mlp(feature_rgb_rotate)
        q_rotate = self.rgb_head(feature_rgb_rotate.reshape(-1, feature_rgb_rotate.shape[2]))


        feature_xyz, feature_rgb = self.feature_fusion(xyz_feature, rgb_feature)
        feature_rgb_global = feature_rgb[:, 0, :]
        feature_xyz_global = feature_xyz.mean(dim=1)
        feature_rgb = feature_rgb[:, 1:, :]
        q = self.rgb_head(feature_rgb.reshape(-1, feature_rgb.shape[2]))
        k = self.xyz_head(feature_xyz.reshape(-1, feature_xyz.shape[2]))


        xyz_feature = xyz_feature.view(-1, xyz_feature.shape[2])
        rgb_feature = rgb_feature.view(-1, rgb_feature.shape[2])
        patch_no_zeros_indices = torch.nonzero(torch.all(xyz_feature != 0, dim=1))

        lossc = (self.contrastive_loss(q[patch_no_zeros_indices,:].squeeze(), k[patch_no_zeros_indices,:].squeeze()) + \
            self.contrastive_loss(k[patch_no_zeros_indices,:].squeeze(), q[patch_no_zeros_indices,:].squeeze()))/2.0 * 1.0

        index_select = torch.randperm(patch_no_zeros_indices.nelement())[:len(patch_no_zeros_indices)//6]
        losscr = self.contrastive_loss(q[index_select,:].squeeze(), q_rotate[index_select,:].squeeze())*0.05

        lossc+=losscr
        idx_class = []
        for i in range(10):
            idx_per = np.where(np.array(classlabels==i))[0]
            idx_class.append(idx_per)

        xyz_feature = xyz_feature.reshape(bs, -1, 1152)
        self.start_iteration -= 1
        global_feat1, global_feat2 = [], []
        if self.start_iteration <=0:
            lossm, count, lossm_r = 0, 0, 0
            for i in range(feature_rgb.shape[0]):
                if len(self.queue_rgb[classlabels[i]])>=self.banksize-100:
                    queue_class_point = self.queue_point[classlabels[i]]
                    matrix_class_point = torch.cat(queue_class_point, dim=0).reshape(-1, 1152).cuda()
                    nonzero_index = torch.nonzero(torch.all(xyz_feature[i] != 0, dim=1))
                    dist_per_point = euclidean_dist(feature_xyz[i][nonzero_index, :].squeeze(1), matrix_class_point)
                    mindist_point = torch.min(dist_per_point, dim=1)
                    maxdist_point = torch.max(dist_per_point, dim=1)
                    queue_class = self.queue_rgb[classlabels[i]]
                    matrix_class = torch.cat(queue_class, dim=0).reshape(-1, 768).cuda()
                    dist_per = euclidean_dist(feature_rgb[i][:, :].squeeze(1), matrix_class)
                    mindist = torch.min(dist_per, dim=1)
                    maxdist = torch.max(dist_per, dim=1)
                    global_cur_rgb, global_cur_point = feature_rgb[i][nonzero_index, :].squeeze(1).mean(dim=0), feature_xyz[i][nonzero_index, :].squeeze(1).mean(dim=0)
                    queue_class_global = self.queue_rgb_global[classlabels[i]]
                    matrix_class_global = torch.cat(queue_class_global, dim=0).reshape(-1, 768).cuda()
                    dist_per_global = euclidean_dist(global_cur_rgb.unsqueeze(0), matrix_class_global)
                    mindist_global = torch.min(dist_per_global, dim=1)

                    queue_class_point_global= self.queue_point_global[classlabels[i]]
                    matrix_class_point_global = torch.cat(queue_class_point_global, dim=0).reshape(-1, 1152).cuda()
                    dist_per_point_global = euclidean_dist(global_cur_point.unsqueeze(0), matrix_class_point_global)
                    mindist_point_global = torch.min(dist_per_point_global, dim=1)

                    lossm+= (mindist[0].mean() + mindist_point[0].mean()) + (mindist_global[0].mean()+mindist_point_global[0].mean()) * 0.4
                    count+=1.0


                    global_feat1.append(global_cur_rgb)
                    global_feat2.append(global_cur_point)
            if count!=0:
                lossm/=count
                lossc += lossm*0.9

        global_feat1, global_feat2 = torch.cat(global_feat1, dim=0), torch.cat(global_feat2, dim=0)
        global_feat1, global_feat2 = self.rgb_head(global_feat1.reshape(-1, 768)), self.xyz_head(global_feat1.reshape(-1, 1152))
        lg = self.contrastive_loss(global_feat1.squeeze(), global_feat2.squeeze()) * 0.1
        lossc+=lg
        xyz_feature = xyz_feature.reshape(bs, -1, 1152)
        for i in range(10):
            if len(idx_class[i])!=0:
                
                for j in range(len(idx_class[i])):
                    nonzero_index = torch.nonzero(torch.all(xyz_feature[idx_class[i]][j] != 0, dim=1))
                    self.queue_point[i].extend(feature_xyz[idx_class[i][j]].reshape(-1, 1152)[nonzero_index, :].detach().cpu())
                    self.queue_rgb[i].extend(feature_rgb[idx_class[i][j]].reshape(-1, 768).detach().cpu())

                    self.queue_rgb_global[i].extend(feature_rgb[idx_class[i][j]].reshape(-1, 768)[nonzero_index, :].mean(dim=0).detach().cpu())
                    self.queue_point_global[i].extend(feature_xyz[idx_class[i][j]].reshape(-1, 1152)[nonzero_index, :].mean(dim=0).detach().cpu())

            if len(self.queue_rgb[i])>self.banksize:
                del self.queue_rgb[i][:len(idx_class[i])-self.banksize-1]
                del self.queue_point[i][:len(idx_class[i])-self.banksize-1]  
                del self.queue_rgb_global[i][:len(idx_class[i])-100-1]
                del self.queue_point_global[i][:len(idx_class[i])-100-1] 
        return lossc


from models.pointnet2_utils import interpolating_points
class Model(torch.nn.Module):
    def __init__(self, rgb_backbone_name='vit_base_patch8_224_dino', out_indices=None, checkpoint_path='',
                 pool_last=False, xyz_backbone_name='Point_MAE', group_size=128, num_group=1024, classname = 'bagel'):
        super().__init__()
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        ## RGB backbone
        self.rgb_backbone = timm.create_model(model_name=rgb_backbone_name, pretrained=True,
                                              checkpoint_path=checkpoint_path,
                                              **kwargs)
        self.pool = torch.nn.AdaptiveAvgPool2d((56, 56))
        self.rgb_backbone = self.rgb_backbone.cuda()
    def forward_rgb_features(self, x):
        x = self.rgb_backbone.patch_embed(x)
        x = self.rgb_backbone._pos_embed(x)
        x = self.rgb_backbone.norm_pre(x)
        if self.rgb_backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.rgb_backbone.blocks(x)
        x = self.rgb_backbone.norm(x)

        feat = x[:, 1:].permute(0, 2, 1).view(x.shape[0], -1, 28, 28)
        feat = self.pool(feat)
        return feat

    def forward(self, rgb, xyz=None):
        self.rgb_backbone.eval()
        rgb_features = self.forward_rgb_features(rgb)
        return rgb_features


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')


    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=1.5e-6,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.002, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='/youtu_fuxi_team1_ceph/v_imyuewang/AD/ssl_outputv2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--classname', default='0',type=int,
                        help='url used to set up distributed training')
    parser.add_argument('--savename', default='0.pkl',type=str,
                        help='url used to set up distributed training')
    return parser


from loss import CompactnessLoss, EWCLoss
def get_all_feature(model, data_loader):
    train_feature_space = []
    with torch.no_grad():
        for data_iter_step, (samples, _) in enumerate(data_loader):
            imgs = samples[0].cuda()
            _,_,feats = model(imgs)
            train_feature_space.append(feats)
        train_feature_space = torch.cat(train_feature_space, dim=0).contiguous().cpu().numpy()
    return train_feature_space
from torch.nn.parallel import DataParallel
def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    classall = [
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
    class_name_id = args.classname


    class_name = classall[class_name_id]
    
    dataset_train = data.mvtec3d.MVTec3DTrainFeaturesALL(class_name = class_name, img_size = 224)

    print(dataset_train)
    print('./savemodel/'+args.savename)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        drop_last=True,
    )
    

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    model = FeatureFusionBlock().cuda()
    model.train()

    model_backbone = Model().cuda()
    model_backbone.eval()
    # following timm: set wd as 0 for bias and norm layers
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    model.train()
    for epoch in range(args.start_epoch, 2):
        
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,group_divider=None,model_backbone=model_backbone, 
            args=args
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}
        torch.save(model.state_dict(), './savemodelnew/epoch'+str(epoch)+args.savename)
        #break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

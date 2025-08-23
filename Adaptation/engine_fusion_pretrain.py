import math
import sys
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from torch.autograd import Variable
from data import data_utils as d_utils
import torchvision.transforms as transforms
import random 
from utils.mvtec3d_util import *
def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,group_divider=None,model_backbone=None, 
                    args=None):
    iteration_stop = 100
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = samples[0].cuda()
        batch_size = imgs.shape[0]
        img_feats = samples[-2].cuda()
        point_feats = samples[-3].cuda()
        #imgs_rotate = samples[2].cuda()
        #point_feats_rotate90 = samples[5].cuda()
        #point_feats_rotate180 = samples[6].cuda()
        #point_feats_rotate270 = samples[7].cuda()
        #img_rotate180, img_rotate270 = samples[3].cuda(), samples[4].cuda()

        #point_fliph = samples[8].cuda()
        #point_flipv = samples[9].cuda()
        #img_fliph = samples[10].cuda()
        #img_flipv = samples[11].cuda()
        #img_patch_rotate = samples[-5].cuda()
        #point_patch_rotate = samples[-6].cuda()
        #random_noise_img = torch.rand(img_feats.shape).cuda()
        #random_noise_point = torch.rand(point_feats.shape).cuda()
        #noise_sample_img = img_feats.clone()*0.7 + random_noise_img*0.3
        #noise_sample_point = point_feats.clone()*0.7 + random_noise_point*0.3
        """
        idx = random.randint(0,2)
        idx = 0
        imgrotateall = []
        imgrotateall.append(imgs_rotate)
        imgrotateall.append(img_rotate180)
        imgrotateall.append(img_rotate270)
        imgs_rotate = imgrotateall[idx]

        #idx = random.randint(0,2)
        imgrotateall = []
        pointrotateall = []
        pointrotateall.append(point_feats_rotate90)
        pointrotateall.append(point_feats_rotate180)
        pointrotateall.append(point_feats_rotate270)
        point_feats_rotate = pointrotateall[idx]
        """
        classlabels = samples[-4]
        region_xyz_patch1, region_xyz_patch2, region_rgb_patch1, region_rgb_patch2, img_rotate = samples[-1]
        region_xyz_patch1, region_xyz_patch2, region_rgb_patch1, region_rgb_patch2 = region_xyz_patch1.cuda(), region_xyz_patch2.cuda(), region_rgb_patch1.cuda(), region_rgb_patch2.cuda()
        with torch.no_grad():
            img_rotate = img_rotate.cuda()
            img_feats_rotate = model_backbone(img_rotate)
            img_feats_rotate = img_feats_rotate.reshape(img_feats_rotate.shape[0], -1, img_feats_rotate.shape[1])
            #imgs_flip = torch.cat([img_fliph, img_flipv],dim=0)
            #img_feats_flip = model_backbone(imgs_flip)
            #img_feats_flip = img_feats_flip.reshape(img_feats_flip.shape[0], -1, img_feats_flip.shape[1])
            #point_feats_flip = torch.cat([point_fliph, point_flipv], dim=0)
            #imgs_rotate = torch.cat([imgs, imgs], dim=0)
            #imgs_rotate = model_backbone(imgs_rotate)
            img_feats = img_feats#imgs_rotate.reshape(imgs_rotate.shape[0], -1, imgs_rotate.shape[1])
            point_feats = point_feats#torch.cat([point_feats, point_patch_rotate], dim=0)
            #img_feats_rotate = img_feats_flip
            #point_feats_rotate = point_feats_flip
            #imgs_rotate = torch.cat([imgs_rotate, img_rotate180],dim=0)
            #img_feats_rotate = model_backbone(imgs_rotate)
            #img_feats_rotate = img_feats_rotate.reshape(img_feats_rotate.shape[0], -1, img_feats_rotate.shape[1])
            #point_feats_rotate = torch.cat([point_feats_rotate90, point_feats_rotate180], dim=0)


            #img_feats_rotate = model_backbone(imgs_rotate)
            #img_feats_rotate = img_feats_rotate.reshape(img_feats_rotate.shape[0], -1, img_feats_rotate.shape[1])
            #img_feats_rotate_180 = model_backbone(img_rotate180)
            #img_feats_rotate_180 = img_feats_rotate_180.reshape(img_feats_rotate_180.shape[0], -1, img_feats_rotate_180.shape[1])

            #img_feats_rotate_270 = model_backbone(img_rotate270)
            #img_feats_rotate_270 = img_feats_rotate_270.reshape(img_feats_rotate_270.shape[0], -1, img_feats_rotate_270.shape[1])

            #img_feats_rotate = torch.cat([img_feats_rotate, img_feats_rotate_180, img_feats_rotate_270], dim=0)
            #point_feats_rotate = torch.cat([point_feats_rotate90, point_feats_rotate180, point_feats_rotate270], dim=0).cuda()
            #point_feats_rotate = point_feats_rotate90

        with torch.cuda.amp.autocast():
            loss = model(point_feats, region_xyz_patch1, region_xyz_patch2, img_feats, region_rgb_patch1, region_rgb_patch2, img_feats_rotate, imgs, 0, classlabels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss.float(), optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        optimizer.zero_grad()
        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)

        iteration_stop-=1
        #if iteration_stop==0 and epoch==1:
        #    torch.save(model.state_dict(), './savemodelnew/epochiteration600_'+str(epoch)+args.savename)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
## Setup

We implement this repo with the following environment:
- Python 3.8
- Pytorch 1.9.0
- CUDA 11.3

Install the other package via:

``` bash
pip install -r requirement.txt
```

## Data Download and Preprocess

### Dataset

The `MVTec-3D AD` dataset can be download from the [Official Website](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad). After download, put the dataset in `dataset` folder.

### Checkpoints

The following table lists the pretrain model used in LSFA:

| Backbone | Pretrain Method |
|----------| --------------- |
| Point Transformer | [Point-MAE](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth) |
| Point Transformer | [Point-Bert](https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1) |
| ViT-b/8 | [DINO](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth) |
| ViT-b/8 | [Supervised ImageNet 1K](https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz) |
| ViT-b/8 | [Supervised ImageNet 21K](https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz) |
| ViT-s/8 | [DINO](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth) |

Put the checkpoint files in `checkpoints` folder.


## 1.Extract features for adaptation
cd FeatureExtract
bash pretrain_both.sh


## 2.Train and Test
The extracted point cloud features and RGB features should be placed at './dataset/mvtec3d_preprocessed'
Then run the following instruction to adapt the features and save the adaptors:

cd Adaptation

python3 fusion_pretrain.py    --accum_iter 16 --lr 0.0003 --batch_size 8 --output_dir ./ssl_outputv2 --classname 0


## 3.Run patchcore with adapted features
Use weight from the former step for patchcore:

cd PatchCore

python3 main.py --method_name DINO+Point_MAE --memory_bank multiple --rgb_backbone_name vit_base_patch8_224_dino --xyz_backbone_name Point_MAE --classname 0 --weightpath [saved_weight]


## Thanks

Our repo is built on [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS), [MoCo-v3](https://github.com/facebookresearch/moco-v3) and [M3DM](https://github.com/nomewang/M3DM), thanks their extraordinary works!
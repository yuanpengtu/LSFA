import torch
from tqdm import tqdm

from data.mvtec3d import get_data_loader
from feature_extractors.dino_features import RGBFeatures
from feature_extractors.dino_pointmae_features import RGBPointFeatures
from feature_extractors.dino_pointmae_features_double_lib import DoubleRGBPointFeatures
from feature_extractors.dino_pointmae_features_fusion import FusionFeatures
from feature_extractors.dino_pointmae_features_triple_lib import TripleFeatures
from feature_extractors.pointmae_features import PointFeatures


class M3DM():
    def __init__(self, args, image_size=224):
        self.args = args
        self.image_size = image_size
        self.count = args.max_sample
        if args.method_name == 'DINO':
            self.methods = {
                "DINO": RGBFeatures(args),
            }
        elif args.method_name == 'Point_MAE':
            self.methods = {
                "Point_MAE": PointFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE+cat':
            self.methods = {
                "DINO+Point_MAE+cat": RGBPointFeatures(args),
            }
        elif args.method_name == 'Fusion':
            self.methods = {
                "Fusion": FusionFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE':
            self.methods = {
                "DINO+Point_MAE": DoubleRGBPointFeatures(args),
            }
        elif args.method_name == 'DINO+Point_MAE+Fusion':
            self.methods = {
                "DINO+Point_MAE+Fusion": TripleFeatures(args),
            }

    def fit(self, class_name):
        train_loader = get_data_loader("train", class_name=class_name, img_size=self.image_size)

        flag = 0
        for sample, _ in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            for method in self.methods.values():
                if self.args.save_feature:
                    method.add_sample_to_mem_bank(sample, class_name=class_name)
                else:
                    method.add_sample_to_mem_bank(sample)
                flag += 1
            if flag > self.count:
                flag = 0
                break

        for method_name, method in self.methods.items():
            print(f'\n\nRunning coreset for {method_name} on class {class_name}...')
            method.run_coreset()

        if self.args.memory_bank == 'multiple':
            flag = 0
            for sample, _ in tqdm(train_loader, desc=f'Extracting training set score for class {class_name}'):
                for method_name, method in self.methods.items():
                    print(f'\n\nRunning late fusion for {method_name} on class {class_name}...')
                    method.add_sample_to_late_fusion_mem_bank(sample)
                    flag += 1
                if flag > self.count:
                    flag = 0
                    break

            for method_name, method in self.methods.items():
                print(f'\n\nTraining Dicision Layer Fusion for {method_name} on class {class_name}...')
                method.run_late_fusion()

    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = get_data_loader("test", class_name=class_name, img_size=self.image_size)
        path_list = []
        with torch.no_grad():
            for sample, mask, label, rgb_path in tqdm(test_loader,
                                                      desc=f'Extracting test features for class {class_name}'):
                for method in self.methods.values():
                    method.predict(sample, mask, label)
                    path_list.append(rgb_path)

        for method_name, method in self.methods.items():
            method.calculate_metrics()
            image_rocaucs[method_name] = round(method.image_rocauc, 3)
            pixel_rocaucs[method_name] = round(method.pixel_rocauc, 3)
            au_pros[method_name] = round(method.au_pro, 3)
            print(
                f'Class: {class_name}, {method_name} Image ROCAUC: {image_rocaucs[method_name]:.3f}, {method_name} Pixel ROCAUC: {pixel_rocaucs[method_name]:.3f}, {method_name} AU-PRO: {au_pros[method_name]:.3f}')
            method.save_prediction_maps('./pred_maps', path_list)

        return image_rocaucs, pixel_rocaucs, au_pros

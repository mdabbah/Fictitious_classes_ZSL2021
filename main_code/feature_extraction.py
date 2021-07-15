#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:03:15 2019

@author: war-machince
"""

import os, sys
from torchvision.datasets.cifar import CIFAR10

from misc.backbones import build_backbone
from misc.project_paths import BASE_DIR, FT_DIR, DATASET_DIR

import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models.resnet as resnet_models
import torchvision.models.densenet as densenet_models
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
from global_setting import NFS_path


class CustomedDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_name, transform=None):
        self.dataset_name = dataset_name

        img_dir = os.path.join(NFS_path, f'dataset/{dataset_name}/')
        file_paths = os.path.join(NFS_path, f'dataset/xlsa17/data/{dataset_name}/res101.mat')

        self.matcontent = sio.loadmat(file_paths)
        self.image_files = np.squeeze(self.matcontent['image_files'])
        self.img_dir = img_dir
        self.transform = transform

        self.split_idx = {'CUB': 7, 'AWA2': 5, 'SVHN': 0, 'SUN': 7}[self.dataset_name]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        image_file = os.path.join(self.img_dir,
                                  '/'.join(image_file.split('/')[self.split_idx:]))
        if self.dataset_name == 'SVHN':
            image_file = self.image_files[idx].strip()

        image = self.transform(image_file)
        return image


# %%
input_size = (224, 224)
data_transforms = transforms.Compose([
    transforms.Lambda(lambda path: Image.open(path)),
    transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode == 'RGBA' else img),
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda image_tnsr: image_tnsr.repeat([3, 1, 1]) if image_tnsr.shape[0] == 1 else image_tnsr),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main():
    idx_GPU = 1
    is_save = True
    attribute_type = 'bert'
    datasets = ['SUN']

    feature_types = [ 'densenet_201_D2', 'densenet_201_D3', 'densenet_201_D4', 'densenet_201_T1',
                     'densenet_201_T2', 'densenet_201_T3', 'resnet_101_L1', 'resnet_101_L2', 'resnet_101_L3',
                     'resnet_101_L4']

    feature_types = ['densenet_201_T3', 'resnet_101_L3', 'resnet_101_L4']

    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    for dataset_name in datasets:
        dataset = CustomedDataset(dataset_name, data_transforms)

        batch_size = 96
        dataset_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size, shuffle=False,
                                                     num_workers=4)

        for feature_type in feature_types:
            all_features = []

            print(f' extracting {feature_type} for {dataset_name}')

            extractor = build_backbone(feature_type, idx_end=feature_type.split('_')[2])
            extractor.float().to(device)
            extractor.eval()
            num_batches = len(dataset_loader)
            for i_batch, imgs in enumerate(dataset_loader):
                print(f'{i_batch}/{num_batches}')
                imgs = imgs.to(device)
                with torch.no_grad():
                    features = extractor(imgs)
                all_features.append(features.cpu().numpy())

            all_features = np.concatenate(all_features, axis=0)
            print(f'shape is {all_features.shape}')
            matcontent = dataset.matcontent
            labels = matcontent['labels'].astype(int).squeeze() - 1

            split_path = os.path.join('../', f'dataset/xlsa17/data/{dataset_name}/att_splits.mat')
            matcontent = sio.loadmat(split_path)
            trainval_loc = matcontent['trainval_loc'].squeeze() - 1
            test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
            test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
            att = matcontent['att'].T
            original_att = matcontent['original_att'].T

            print('save w2v_att')
            save_path = os.path.join(BASE_DIR,
                                     f'cvpr20_DAZLE/data/new_extracted/{dataset_name}/feature_map_{feature_type}_{attribute_type}_{dataset_name}.hdf5')
            save_dir = os.path.dirname(save_path)
            os.makedirs(save_dir, exist_ok=True)

            print('-'*60)
            print(f' starting to save {feature_type} for {dataset_name}')
            print(save_path)
            print('-' * 60)
            if is_save:
                f = h5py.File(save_path, "w")
                f.create_dataset('feature_map', data=all_features, compression="gzip")
                f.create_dataset('labels', data=labels, compression="gzip")
                f.create_dataset('trainval_loc', data=trainval_loc, compression="gzip")
                # f.create_dataset('train_loc', data=train_loc, compression="gzip")
                # f.create_dataset('val_unseen_loc', data=val_unseen_loc, compression="gzip")
                f.create_dataset('test_seen_loc', data=test_seen_loc, compression="gzip")
                f.create_dataset('test_unseen_loc', data=test_unseen_loc, compression="gzip")
                # f.create_dataset('att', data=att, compression="gzip")
                # f.create_dataset('original_att', data=original_att, compression="gzip")
                # f.create_dataset('w2v_att', data=w2v_att, compression="gzip")
                f.close()


if __name__ == '__main__':
    main()

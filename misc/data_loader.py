import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from misc.data_utils import load_xlsa17, load_pickle
import numpy as np
import torch.nn.functional as F

from misc.project_paths import DATASET_DIR

supported_datasets = {'CUB', 'AWA1', 'AWA2', 'SUN'}


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort


def get_rand_gaussian_blur(sigma_low=0.1, sigma_high=2.0, kernel_size=24):
    meshgrids = torch.meshgrid([
        torch.arange(kernel_size, dtype=torch.float32),
        torch.arange(kernel_size, dtype=torch.float32)
    ])
    kernel_size = [kernel_size] * 2

    def gaussian_blur(img):
        kernel = 1
        sigma = [np.random.uniform(sigma_low, sigma_high)] * 2
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * np.sqrt(2 * np.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(3, *[1] * (kernel.dim() - 1))
        img = img.unsqueeze(0)
        img = F.conv2d(img, kernel, groups=3)
        return img.squeeze()

    return gaussian_blur


class ZSLDatasetImages(Dataset):
    """common ZSL datasets."""

    def __init__(self, dataset_name: str, features_type: str, ds_subset: str, batch_size: int, split_id: int = 0,
                 color_distortion_strength=1.0, return_augmentations=True, return_batches=False):
        """
        dataset for class for common ZSL datasets with augmentations
        :param dataset_name: one of {'CUB', 'AWA1', 'AWA2', 'SUN'}
        :param features_type: one of {'resnet', 'densenet121', 'densenet201'}
        :param ds_subset: one of {'train', 'valid', 'test_unseen', 'test_seen'}
        :param batch_size: batch size
        """
        if dataset_name not in supported_datasets:
            raise ValueError("supported datasets are 'CUB', 'AWA1', 'AWA2', 'SUN' ")

        if ds_subset in ['all_train', 'test_unseen', 'test_seen']:
            ds_subset += '_loc'
        else:
            ds_subset += f'_loc{split_id}'

        if ds_subset not in ['all_train_loc', 'test_unseen_loc', 'test_seen_loc', f'val_unseen_loc{split_id}',
                             f'val_seen_loc{split_id}', f'train_zsl_loc{split_id}', f'train_gzsl_loc{split_id}'] \
                or not 0 <= split_id < 4:
            raise ValueError(f'unsupported dataset subset {ds_subset}')

        self.dataset_name = dataset_name
        self.features_type = features_type
        self.ds_subset = ds_subset
        self.return_augmentations = return_augmentations
        self.return_batches = return_batches
        self.image_files, self.labels, self.class_attributes = load_xlsa17(dataset=dataset_name, data_subset=ds_subset,
                                                                           features_type=features_type,
                                                                           return_image_paths=True)

        self.norm_paths()

        self.num_samples = len(self.labels)
        self.classes = np.unique(self.labels)

        self.input_size = (224, 224)
        self.color_distortion_strength = color_distortion_strength
        self.transform = transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path)),
            transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
            transforms.RandomResizedCrop(self.input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            get_color_distortion(s=color_distortion_strength),
            transforms.ToTensor()
            # get_rand_gaussian_blur(),
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.transform_open = transforms.Compose([
            transforms.Lambda(lambda path: Image.open(path)),
            transforms.Lambda(lambda image: image.convert('RGB') if image.mode == 'L' else image),
            transforms.Resize(self.input_size),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor()
            , transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.batch_size = batch_size

    def update_labels_and_classes(self, classes):
        self.labels = np.argmax(classes[np.newaxis, :] == self.labels[:, np.newaxis], 1)
        self.classes = np.unique(self.labels)

    def norm_paths(self):
        img_dir = '../dataset/CUB/'
        if self.dataset_name == 'CUB':

            def convert_path(image_files, img_dir):
                new_image_files = []
                for idx in range(len(image_files)):
                    image_file = image_files[idx][0]
                    image_file = os.path.join(img_dir, '/'.join(image_file.split('/')[7:]))
                    new_image_files.append(image_file)
                return np.array(new_image_files)

            self.image_files = convert_path(self.image_files, img_dir)

    def __len__(self):
        """
        :return: number of batches
        """
        return len(self.image_files) if not self.return_batches else len(self.image_files) // self.batch_size + 1

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, idx):
        """
        :param idx: batch index
        :return: idx-th batch (features, labels, attributes) first dim is batch size (#samples in batch)
        """
        if idx >= len(self):
            raise IndexError

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.return_batches:
            imgs_pths = self.image_files[idx * self.batch_size: (idx + 1) * self.batch_size, :]
            Y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
            X = [self.transform_open(img_pth) for img_pth in imgs_pths]
            sample = torch.stack(X).to(self.device), torch.tensor(Y).to(self.device)

            return sample

        img_pth = self.image_files[idx]
        imgs = self.transform_open(img_pth)
        if self.return_augmentations:
            img_aug = self.transform(img_pth)
            imgs = torch.stack([imgs, img_aug], dim=0)

        labels = torch.tensor([self.labels[idx]]).squeeze()

        return imgs, labels

    def next_batch(self, batch_size=None, simclr_duplicate=False, perm=None):
        batch_size = batch_size if batch_size else self.batch_size

        if perm is None:
            perm = torch.randperm(self.num_samples)[0:batch_size]

        batch_images = []
        for index in perm:
            img = self.image_files[index]
            batch_images.append(self.transform(img))

        if simclr_duplicate:
            for index in perm:
                img = self.image_files[index]
                batch_images.append(self.transform_open(img))

        batch_images = torch.stack(batch_images).to(self.device)

        with torch.no_grad():
            batch_feature = self.backbone(batch_images)

        batch_label = torch.LongTensor(self.labels[perm])

        batch_att = torch.tensor(self.class_attributes[batch_label])

        return batch_label.to(self.device), batch_feature.float(), \
               batch_att.to(self.device).float(), perm


def collate_aug(samples):
    # imgs = [s[0] for s in samples]
    regular = [s[0][0] for i, s in enumerate(samples)]
    augmented = [s[0][1] for i, s in enumerate(samples)]
    imgs = regular + augmented
    labels = [s[1] for i, s in enumerate(samples)]
    return torch.stack(imgs, 0), torch.stack(labels, 0).long()


class ZSLDatasetEmbeddings(Dataset):
    """common ZSL datasets."""

    def __init__(self, dataset_name: str, features_type: str, ds_subset: str, batch_size: int, split_id: int = 0,
                 conv_features: bool = False, offset=-1, shuffle_batch=False, balance_dataset=False):
        """
        dataset for class for common ZSL datasets
        :param dataset_name: one of {'CUB', 'AWA1', 'AWA2', 'SUN'}
        :param features_type: one of {'resnet', 'densenet121', 'densenet201'}
        :param ds_subset: one of {'train', 'valid', 'test_unseen', 'test_seen'}
        :param batch_size: batch size
        """
        # if dataset_name not in supported_datasets:
        #     raise ValueError("supported datasets are 'CUB', 'AWA1', 'AWA2', 'SUN' ")

        #  validation keywords are: train_gzsl_loc{fold_id}, f'val_seen_loc{fold_id}', f'val_unseen_loc{fold_id}'

        if ds_subset in ['all_train', 'test_unseen', 'test_seen', 'supervised_train', 'supervised_test',
                         'orig_supervised_train', 'orig_supervised_test', 'unseen_te', 'unseen_tr']:
            ds_subset += '_loc'
        else:
            ds_subset += f'_loc{split_id}'

        self.dataset_name = dataset_name
        self.features_type = features_type
        self.ds_subset = ds_subset
        self.features, self.labels, self.class_attributes = load_xlsa17(dataset=dataset_name, data_subset=ds_subset,
                                                                        features_type=features_type,
                                                                        conv_features=conv_features)
        self.num_samples = len(self.labels)
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        self.offset = offset
        if offset >= 0:
            classes, self.labels = np.unique(self.labels, return_inverse=True)
            self.classes = np.arange(len(classes))
            self.labels += offset
            self.classes += offset

        self.balance_dataset = balance_dataset
        if balance_dataset:
            _, labels = np.unique(self.labels, return_inverse=True)
            weights = 1. / np.unique(labels, return_counts=True)[1]
            self.weights = torch.from_numpy(weights[labels])
            self.class_idxs = self.get_idx_classes()

        self.shuffle_batch = shuffle_batch

        self.transform = transforms.ToTensor()
        self.batch_size = batch_size

    def consume(self, other):

        self.features = np.concatenate([self.features, other.features])
        self.labels = np.concatenate([self.labels, other.labels])
        self.num_samples = len(self.labels)
        self.classes = np.unique(self.labels)

    def update_labels_and_classes(self, classes):
        self.labels = np.argmax(classes[np.newaxis, :] == self.labels[:, np.newaxis], 1)
        self.classes = np.unique(self.labels)
        self.class_attributes = self.class_attributes[classes]

    def redefine_classes(self, start_class):
        label_set, labels = np.unique(self.labels, return_inverse=True)
        self.labels = labels + start_class
        self.classes = np.unique(self.labels)

    def get_class_attributes(self):
        labels = np.unique(self.labels)
        return np.copy(self.class_attributes[labels]), labels

    def get_data(self):
        return np.copy(self.features), np.copy(self.labels)

    def __len__(self):
        """
        :return: number of batches
        """
        return (self.features.shape[0] // self.batch_size) + 1

    def to(self, device):
        self.device = device
        return self

    def shuffle(self):
        perm = np.random.permutation(len(self.labels))
        self.features = self.features[perm]
        self.labels = self.labels[perm]

    def __getitem__(self, idx):
        """
        :param idx: batch index
        :return: idx-th batch (features, labels, attributes) first dim is batch size (#samples in batch)
        """
        if idx >= len(self):
            if not self.balance_dataset and self.shuffle_batch:
                self.shuffle()
            raise IndexError

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.shuffle_batch:
            labels, features = self.next_batch(return_atts=False)
            return features, labels

        X = self.features[idx * self.batch_size: (idx + 1) * self.batch_size, :]
        Y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        sample = torch.tensor(X), torch.tensor(Y)

        return sample

    def get_idx_classes(self):
        n_classes = len(self.classes)
        self.idxs_list = []
        membership_matrix = self.classes[:, np.newaxis] == self.labels[np.newaxis, :]
        for i in range(n_classes):
            idx_c = np.argwhere(membership_matrix[i]).squeeze()
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def pick_balanced_batch(self):
        idx = []
        n_samples_class = max(self.batch_size // self.num_classes, 1)
        sampled_idx_c = np.random.choice(np.arange(self.num_classes), min(self.num_classes, self.batch_size),
                                         replace=False).tolist()
        for i_c in sampled_idx_c:
            idxs = self.idxs_list[i_c]
            idx.append(np.random.choice(idxs, n_samples_class))
        idx = np.concatenate(idx)
        idx = torch.from_numpy(idx)
        return idx

    def next_batch(self, return_perm=False, return_atts=True):

        if self.balance_dataset:
            idx = torch.multinomial(self.weights, self.batch_size, replacement=True)
            # idx = self.pick_balanced_batch()
        else:
            idx = torch.randperm(self.num_samples)[0:self.batch_size]

        batch_feature = torch.tensor(self.features[idx])

        batch_label = torch.LongTensor(self.labels[idx])

        if not return_atts:
            return batch_label, batch_feature.float()

        if self.offset >= 0:
            batch_att = torch.tensor(self.class_attributes[batch_label-self.offset])

        if not return_perm:
            return batch_label, batch_feature.float(), \
                   batch_att.float()

        return batch_label, batch_feature.float(), \
               batch_att.float(), idx


class ZSLDatasetDescriptions(Dataset):
    """common ZSL datasets."""

    def __init__(self, dataset_name: str, ds_subset: str, batch_size: int, ):
        """
        dataset for class for {CUB, FLO} containing the lines describing samples of
        {CUB, FLO} gathered by https://arxiv.org/pdf/1605.05395.pdf
        :param dataset_name: one of {'CUB', 'FLO'}
        :param ds_subset: one of {'train', 'test'}
        :param batch_size: batch size
        """
        self.dataset_name = dataset_name
        self.ds_subset = ds_subset

        supported_datasets = {'CUB'}
        if dataset_name not in supported_datasets:
            raise ValueError("supported datasets are ONLY 'CUB' at the moment")

        if ds_subset not in ['train', 'test']:
            ValueError("supported subsets are ONLY {'train', 'test'} at the moment")

        save_dir = os.path.join(DATASET_DIR, 'cub_desc_dataset')
        path = os.path.join(save_dir, f'{ds_subset}_subset.pkl')
        data = load_pickle(path)

        self.samples = np.array(data[f'{ds_subset}_lines'])
        self.labels = np.array(data[f'{ds_subset}_labels'])

        self.num_samples = len(self.labels)
        self.shuffle()

        self.classes = np.unique(self.labels)

        self.batch_size = batch_size

    def __len__(self):
        """
        :return: number of batches
        """
        return (self.samples.shape[0] // self.batch_size) + 1

    def __getitem__(self, idx):
        """
        :param idx: batch index
        :return: idx-th batch (features, labels, attributes) first dim is batch size (#samples in batch)
        """
        if idx >= len(self):
            raise IndexError

        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.samples[idx * self.batch_size: (idx + 1) * self.batch_size]
        Y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        sample = X.tolist(), Y.tolist()
        return sample

    def next_batch(self, return_perm=False):

        idx = torch.randperm(self.num_samples)[0:self.batch_size]

        batch_samples = self.samples[idx]

        batch_label = self.labels[idx]

        return batch_samples.tolist(), batch_label.tolist()

    def shuffle(self):
        perm = np.random.permutation(self.num_samples)

        self.samples = self.samples[perm]
        self.labels = self.labels[perm]


if __name__ == '__main__':
    ds_loader = ZSLDatasetImages('CUB', 'densenet_201_T3', 'train', 16).to(torch.device('cuda:0'))
    ds_loader2 = ZSLDatasetEmbeddings('CUB', 'densenet_201_T3', 'train', 16, conv_features=True).to(
        torch.device('cuda:0'))
    batch = ds_loader.next_batch()
    bat = ds_loader[5]

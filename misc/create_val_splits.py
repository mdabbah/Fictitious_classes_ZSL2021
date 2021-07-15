import os

from scipy.io import loadmat
import numpy as np

from misc.data_utils import load_xlsa17, save_pickle, get_class_names, split_dataset, load_xlsa17_img_paths
from misc.project_paths import DATASET_DIR
import pandas as pd


def create_splits(dataset_name):
    att_path = f'../dataset/xlsa17/data/{dataset_name}/att_splits.mat'
    atts_and_splits = loadmat(att_path)
    data_path = f'../dataset/xlsa17/data/{dataset_name}/res101.mat'
    Y = loadmat(data_path)['labels'] - 1

    all_train_loc = atts_and_splits['trainval_loc'] - 1  # matlab starts with 1 python with 0
    test_unseen_loc = atts_and_splits['test_unseen_loc'] - 1  # matlab starts with 1 python with 0
    test_seen_loc = atts_and_splits['test_seen_loc'] - 1  # matlab starts with 1 python with 0
    Y = Y[all_train_loc].squeeze()
    all_classes_names = atts_and_splits['allclasses_names']
    locs = {'all_train_loc': all_train_loc, 'test_unseen_loc': test_unseen_loc, 'test_seen_loc': test_seen_loc}

    for split_id in range(1, 4):
        train_classes_path = f'../dataset/xlsa17/data/{dataset_name}/trainclasses{split_id}.txt'
        with open(train_classes_path) as f:
            train_classes_names = f.readlines()
        train_classes_names = [t[:-1] for t in train_classes_names]

        val_classes_path = f'../dataset/xlsa17/data/{dataset_name}/valclasses{split_id}.txt'
        with open(val_classes_path) as f:
            val_classes_names = f.readlines()
        val_classes_names = [t[:-1] for t in val_classes_names]

        test_classes_path = f'../dataset/xlsa17/data/{dataset_name}/testclasses.txt'
        with open(test_classes_path) as f:
            test_classes_names = f.readlines()
        test_classes_names = [t[:-1] for t in test_classes_names]

        val_classes_numeric = []
        train_classes_numeric = []
        test_classes_numeric = []
        for i, c in enumerate(all_classes_names):
            if c in val_classes_names:
                val_classes_numeric.append(i)

            if c in train_classes_names:
                train_classes_numeric.append(i)

            if c in test_classes_names:
                test_classes_numeric.append(i)

        val_classes_numeric = np.array(val_classes_numeric)
        train_classes_numeric = np.array(train_classes_numeric)
        val_unseen_loc = np.argwhere(Y[:, np.newaxis] == val_classes_numeric)[:, 0]
        train_loc = np.argwhere(Y[:, np.newaxis] == train_classes_numeric)[:, 0]
        test_loc = np.argwhere(Y[:, np.newaxis] == test_classes_numeric)[:, 0]

        assert test_loc.size == 0
        assert np.all(val_classes_numeric[np.unique(np.argwhere(Y[:, np.newaxis] == val_classes_numeric)[:, 1])]
                      == val_classes_numeric)

        assert np.all(train_classes_numeric[np.unique(np.argwhere(Y[:, np.newaxis] == train_classes_numeric)[:, 1])]
                      == train_classes_numeric)

        assert np.intersect1d(train_classes_numeric, val_classes_numeric).size == 0

        train_seen_membership_matrix = Y[:, np.newaxis] == train_classes_numeric

        num_train_classes = train_seen_membership_matrix.shape[1]

        seen_samples_per_class = [np.argwhere(train_seen_membership_matrix[:, i]) for i in range(num_train_classes)]
        num_samples_4_seen_val = 11
        val_seen_loc = np.concatenate([c[:num_samples_4_seen_val].squeeze() for c in seen_samples_per_class])

        train_gzsl = np.concatenate([c[num_samples_4_seen_val:].squeeze() for c in seen_samples_per_class])

        locs.update({f'val_unseen_loc{split_id}': val_unseen_loc,
                     f'val_seen_loc{split_id}': val_seen_loc, f'train_zsl_loc{split_id}': train_loc,
                     f'train_gzsl_loc{split_id}': train_gzsl})

    locs = split_unseen_samples(dataset_name, locs)
    locs = {k: np.squeeze(v) for k, v in locs.items()}
    save_folder = '../dataset/split_info'
    os.makedirs(save_folder, exist_ok=True)
    fpath = os.path.join(save_folder, f'{dataset_name}_split_info.pkl')
    save_pickle(fpath, locs)
    return locs


def create_orig_splits(dataset_name, splits=None):
    if splits is None:
        splits = {}

    samples_paths, labels = load_xlsa17_img_paths(dataset_name)
    samples_paths = np.array([p[0].split('images/')[1] for p in samples_paths.squeeze()])
    fpath = f'../dataset/{dataset_name}/images.txt'
    orig_samples_paths = np.array(pd.read_table(fpath, names=['cid', 'sample_path'], sep=' ')['sample_path']).squeeze()

    mapping_mat = samples_paths[:, np.newaxis] == orig_samples_paths[np.newaxis, :]

    mapping = np.argmax(mapping_mat, axis=1)
    mapping_inv = np.argmax(mapping_mat, axis=0)

    orig_train_test_split = np.array(
        pd.read_table('../dataset/CUB/train_test_split.txt', names=['sid', 'is_train'], sep=' ')
        ['is_train'], dtype=np.bool)

    orig_supervised_train_loc = mapping_inv[orig_train_test_split]
    orig_supervised_test_loc = mapping_inv[np.logical_not(orig_train_test_split)]

    return splits


def split_unseen_samples(dataset_name, splits):
    data_path = f'../dataset/xlsa17/data/{dataset_name}/res101.mat'
    Y = loadmat(data_path)['labels'] - 1
    unseen_classes_sid = splits['test_unseen_loc']
    unseen_classes_sid_labels = Y[unseen_classes_sid]
    tr_sample, te_samples = split_dataset(unseen_classes_sid_labels, percentage=0.5)

    splits['unseen_tr_loc'] = tr_sample
    splits['unseen_te_loc'] = te_samples

    return splits


dataset_name = 'SUN'

sp = create_splits(dataset_name)
# split_unseen_samples(dataset_name, splits=sp)

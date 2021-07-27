import os
import pickle
import pandas as pd
import h5py
from scipy.io import loadmat, savemat
import numpy as np

from misc.project_paths import DATASET_DIR, DATA_BASE_DIR


feature_and_labels_cache = {}


def get_attribute_groups(dataset_name):
    if dataset_name != 'CUB':
        raise ValueError(f'{dataset_name} is not supported')
    atts = pd.read_csv('../dataset/CUB/attributes/attributes.txt', sep='::', engine='python',
                       names=['group', 'value'])

    groups = list(atts['group'])
    groups = np.array([gr.split(' ')[1] for gr in groups])
    unique_groups, idx = np.unique(groups, return_index=True)
    unique_groups = unique_groups[np.argsort(idx)]
    membership_mat = unique_groups[:, np.newaxis] == groups[np.newaxis, :]

    # group2attributes = [[] for i in range(len(unique_groups))]
    # group2id = np.arange(len(unique_groups))
    # seen_groups = []
    # for att_id, att in enumerate(groups):
    #     if att in seen_groups:
    #         group2attributes.append(att_id)
    return membership_mat


def load_features_and_labels(dataset, features_type, conv_features, tag=''):
    if not conv_features:
        features_type = {'resnet101': 'res101', 'densenet121': 'dense121', 'densenet201': 'dense201'}[features_type]
        data_path = f'../dataset/new_version/{dataset}/{features_type}.mat'
        data = loadmat(data_path)

        features = data['features'].T
        labels = data['labels'] - 1  # matlab starts with 1 python with 0
        labels = np.squeeze(labels)
        assert features.shape[0] == len(labels.squeeze())
        return features, labels

    features_type = {'resnet101': 'ResNet_101', 'densenet121': 'densenet_121',
                     'densenet201': 'densenet_201'}.get(features_type, features_type)

    data_path = f'{DATA_BASE_DIR}/precomputed_features/{dataset}/feature_map_{features_type}_bert_{dataset}{tag}.hdf5'
    print(f'loaded features from: {data_path}')
    fl = feature_and_labels_cache.get(data_path, None)
    if fl is None:
        hf = h5py.File(data_path, 'r')
        features = np.array(hf.get('feature_map'))
        labels = np.array(hf.get('labels'))
        feature_and_labels_cache[data_path] = (features, labels)
    else:
        print('found them in cache already! :)')
        features, labels = fl
    return features, np.squeeze(labels)


def load_attributes(dataset_name, attributes_type, features_type=None, split='PS'):
    use_exem = attributes_type.find('EXEM') >= 0
    use_gfzsl = attributes_type.find('gfzsl') != -1
    if use_exem:
        features_type = 'resnet'
        exem_path = f'../dataset/data/EXEM_info/{dataset_name}_{split}_{attributes_type}_{features_type}.mat'
        class_attributes = loadmat(exem_path)['attr2']

    elif use_gfzsl:
        gfzsl_path = f'../dataset/data/gfzsl_info/{dataset_name}_{split}_{attributes_type}_{features_type}.mat'
        class_attributes = loadmat(gfzsl_path)['attr2']

    elif attributes_type == 'original':
        atts_path = f'../dataset/data/{dataset_name}_{split}_resnet.mat'
        atts_and_splits = loadmat(atts_path)
        class_attributes = atts_and_splits['attr2']  # type: np.ndarray #(seen+unseen_classes) X attributes_dim
    else:
        raise ValueError(f"unrecognized attributes type {attributes_type}")

    if (dataset_name.find('SUN') == -1) and (not use_exem) and (not use_gfzsl):
        class_attributes = class_attributes / 100

    return class_attributes


def load_xlsa17_img_paths(dataset_name):
    data_path = f'{DATA_BASE_DIR}/xlsa17/data/{dataset_name}/res101.mat'
    data = loadmat(data_path)
    samples, labels = data['image_files'], data['labels'] - 1  # matlab to python
    return samples, labels


def norm_img_paths(dataset_name, image_files_paths):
    img_dir = f'{DATA_BASE_DIR}/{dataset_name}/'

    def convert_path(image_files, img_dir):
        new_image_files = []
        split_idx = {'CUB': 7, 'AWA2': 5, 'SVHN': 5, 'SUN': 7}[dataset_name]
        for idx in range(len(image_files)):
            image_file = image_files[idx][0] if dataset_name != 'SVHN' else image_files[idx]
            image_file = os.path.join(img_dir, '/'.join(image_file.split('/')[split_idx:])).strip()
            new_image_files.append(image_file)
            assert os.path.isfile(image_file)
        return np.array(new_image_files)

    if dataset_name in ('CUB', 'AWA2', 'SVHN', 'SUN'):
        image_files = convert_path(image_files_paths, img_dir)
    else:
        image_files = np.array([s.strip() for s in image_files_paths])

    return image_files


def load_splits_locs(dataset_name):
    splits_locs_path = f'{DATA_BASE_DIR}/split_info/{dataset_name}_split_info.pkl'
    splits_locs = load_pickle(splits_locs_path)
    return splits_locs


def load_xlsa17(dataset: str, data_subset: str, features_type: str = 'resnet101', conv_features: bool = False,
                return_image_paths: bool = False):
    """

    :param dataset: can be: 'AWA1' 'AWA2' 'CUB' 'SUN'
    :param data_subset: can be: 'train' 'test_unseen' 'test_generalized'
    :param features_type: can be: 'dense201' 'dense121' 'res101'
    :return:
    """

    splits_locs_path = f'{DATA_BASE_DIR}/split_info/{dataset}_split_info.pkl'
    splits_locs = load_pickle(splits_locs_path)

    att_path = f'{DATA_BASE_DIR}/xlsa17/data/{dataset}/att_splits.mat'
    if os.path.isfile(att_path):
        atts_and_splits = loadmat(att_path)

        class_attributes = atts_and_splits['original_att']
        if dataset.find('SUN') == -1 and dataset.find('AWA2') == -1:
            class_attributes = class_attributes / 100
        if 'AWA2' in dataset:
            class_attributes[class_attributes < 0] = 0

    else:
        class_attributes = np.zeros([1, 1])

    if not return_image_paths and features_type is not None:
        samples, labels = load_features_and_labels(dataset, features_type, conv_features)
    else:
        data_path = f'../dataset/xlsa17/data/{dataset}/res101.mat'
        data = loadmat(data_path)
        samples, labels = data['image_files'], np.squeeze(data['labels'] - 1)

    num_class = len(np.unique(labels))
    if class_attributes.shape[0] != num_class:
        class_attributes = class_attributes.T  # make dim 0 be the number of classes

    if data_subset not in ['all_train_loc', 'test_unseen_loc', 'test_seen_loc', 'supervised_train_loc',
                           'supervised_test_loc', 'orig_supervised_train_loc', 'orig_supervised_test_loc',
                           'unseen_tr_loc', 'unseen_te_loc']:
        samples = samples[splits_locs['all_train_loc']].squeeze()
        labels = labels[splits_locs['all_train_loc']].squeeze()

    if (data_subset in ['unseen_tr_loc', 'unseen_te_loc']) and dataset != 'SVHN':
        samples = samples[splits_locs['test_unseen_loc']].squeeze()
        labels = labels[splits_locs['test_unseen_loc']].squeeze()

    samples, labels = samples[splits_locs[data_subset]].squeeze(), labels[splits_locs[data_subset]].squeeze()
    classes = np.unique(labels)
    if class_attributes.shape[0] > 1:
        class_attributes = class_attributes[classes, :]
    return samples, labels, class_attributes


def split_train_val_xlsa17(dataset_name: str, split_id: int, X: np.ndarray, Y: np.ndarray):
    """
    splits the given features and labels to training and validation
    according to the desired split defined by Xian et al. in the good the bad and the ugly
    :param dataset_name: from which dataset were feature and labels taken from
    could be one of {'CUB', 'AWA1', 'AWA2', 'SUN'}
    :param split_id: one of {1, 2, 3}, the split defined by trainClasses{split_id}.txt and valClasses{split_id}.txt
    :param X: those are the features of the samples, shape (#samples, features_dim)
    :param Y: those are the features of the samples, shape (#samples, 1)
    :return: X_train, Y_train, X_val, Y_val
    """

    att_path = f'../dataset/xlsa17/data/{dataset_name}/att_splits.mat'
    atts_and_splits = loadmat(att_path)
    all_classes_names = atts_and_splits['allclasses_names']

    train_classes_path = f'../dataset/xlsa17/data/{dataset_name}/trainclasses{split_id}.txt'
    with open(train_classes_path) as f:
        train_classes_names = f.readlines()
    train_classes_names = [t[:-1] for t in train_classes_names]

    val_classes_path = f'../dataset/xlsa17/data/{dataset_name}/valclasses{split_id}.txt'
    with open(val_classes_path) as f:
        val_classes_names = f.readlines()

    val_classes_names = [t[:-1] for t in val_classes_names]

    val_classes_numeric = []
    train_classes_numeric = []
    for i, c in enumerate(all_classes_names):
        if c in val_classes_names:
            val_classes_numeric.append(i)

        if c in train_classes_names:
            train_classes_numeric.append(i)

    val_loc, _ = np.where(Y[:, np.newaxis] == val_classes_numeric)
    train_loc, _ = np.where(Y[:, np.newaxis] == train_classes_numeric)

    val_loc = np.sort(val_loc)
    train_loc = np.sort(train_loc)

    return X[train_loc, :], Y[train_loc], X[val_loc, :], Y[val_loc]


def load_attribute_descriptors(attribute_desc_type='bert', dataset='CUB', pickle_path=None):
    if not pickle_path:
        pickle_path = os.path.join(DATA_BASE_DIR, f'w2v/{dataset}_attribute_{attribute_desc_type}.pkl')
    with open(pickle_path, 'rb') as f:
        w2v_att = pickle.load(f)
    print(f'loaded w2v from {pickle_path}')
    return w2v_att


def save_attribute_descriptors(attribute_desc, attribute_desc_type='bert', dataset='CUB', pickle_path=None):
    if not pickle_path:
        pickle_path = os.path.join(DATA_BASE_DIR, f'w2v/{dataset}_attribute_{attribute_desc_type}.pkl')
    save_pickle(pickle_path, attribute_desc)
    print(f'saved attributes at {pickle_path}')


def load_classes_paragraphs(w2v_extractor='bert', dataset='CUB', tag='multiple_sent'):
    pkl_path = f'{DATASET_DIR}/class_signatures/{dataset}/{dataset}_{w2v_extractor}_{tag}.pkl'
    with open(pkl_path, 'rb') as f:
        classes_paragraphs = pickle.load(f)
    print(f'loaded class signatures from {pkl_path}')
    return classes_paragraphs


def average_data_frames(files):
    df = pd.read_csv(files[0])

    mean_df = pd.DataFrame(columns=df.columns)
    columns = df.columns
    for col in columns:
        folds_traces = []
        for ffile in files:
            df = pd.read_csv(ffile)
            trace = np.array(df[col])
            folds_traces.append(trace[np.newaxis, :])

        min_len = np.min([t.shape[1] for t in folds_traces])
        folds_traces = [ft[:, :min_len] for ft in folds_traces]
        mean_trace = np.mean(np.concatenate(folds_traces, axis=0), axis=0)
        mean_df[col] = mean_trace

    f_path = files[0].split('fold')[0] + 'mean over folds.csv'
    mean_df.to_csv(f_path, index=False)


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    obj = None
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)

    return obj


def get_class_names(dataset_name, with_numbers=False):
    classes_names_path = f'../dataset/xlsa17/data/{dataset_name}/allclasses.txt'
    with open(classes_names_path) as f:
        classes_names = f.readlines()

    if not with_numbers:
        classes_names = [cl.split('.')[1] for cl in classes_names]

    classes_names = [cl[:-1] if cl[-1] == '\n' else cl for cl in classes_names]
    return classes_names


def split_dataset(labels, percentage=0.7, offset=0):
    labels = np.array(labels).squeeze()[:, np.newaxis]
    label_set = np.unique(labels)
    membership_matix = labels == label_set[np.newaxis, :]
    train_locs = []
    test_locs = []
    for cid, class_ in enumerate(label_set):
        samples_inds = np.argwhere(membership_matix[:, cid]).squeeze()
        perm = np.random.permutation(len(samples_inds))
        samples_inds = samples_inds[perm]
        num_training_samples = int(percentage * len(samples_inds))
        train_locs.append(samples_inds[:num_training_samples])
        test_locs.append(samples_inds[num_training_samples:])

    return np.concatenate(train_locs) + offset, np.concatenate(test_locs) + offset


def redefine_labels(label_set, labels):
    labels = np.argmax(label_set[np.newaxis, :] == labels[:, np.newaxis], 1)
    return labels


def _test_val_splits(dataset):
    for fold in [1, 2, 3]:
        val_seen_imgs, val_seen_labels, a = load_xlsa17(dataset, f'val_seen_loc{fold}', return_image_paths=True)
        val_unseen_imgs, val_unseen_labels, ab = load_xlsa17(dataset, f'val_unseen_loc{fold}', return_image_paths=True)
        tr_ims, tr_labels, ab2 = load_xlsa17(dataset, f'train_gzsl_loc{fold}', return_image_paths=True)

        seen_cls = np.unique(tr_labels)
        s_cls = np.unique(val_seen_labels)
        unseen_cls = np.unique(val_unseen_labels)

        assert len(np.intersect1d(val_seen_imgs, val_unseen_imgs)) == 0
        assert len(np.intersect1d(val_seen_imgs, tr_ims)) == 0
        assert len(np.intersect1d(tr_ims, val_unseen_imgs)) == 0

        assert len(np.intersect1d(seen_cls, unseen_cls)) == 0
        assert len(np.setdiff1d(seen_cls, s_cls)) == 0
        assert len(np.setdiff1d(s_cls, seen_cls)) == 0


if __name__ == '__main__':
    dataset_name = 'CUB'
    _test_val_splits(dataset_name)

    split_id = 3

    X, Y, c = load_xlsa17(dataset_name, 'test')

    X_tr, Y_tr, X_val, Y_val = split_train_val_xlsa17(dataset_name, split_id, X, Y)
    a = 5


def average_over_folds(csv_files):
    read_files = []
    for file in csv_files:
        if file in read_files:
            continue

        f_name = file.split('fold')[0]
        fold_files = [f for f in csv_files if f.find(f_name) >= 0]
        read_files.extend(fold_files)
        average_data_frames(fold_files)

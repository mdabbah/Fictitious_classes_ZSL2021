import functools
import gc
import itertools
import os

import torch
import numpy as np
from scipy.special import softmax
from torch import optim

from dazle_model import Dazle
from misc import data_utils
from misc.custome_dataset import get_base_transforms, get_testing_transform
from misc.data_loader import ZSLDatasetEmbeddings
from misc.data_utils import load_attribute_descriptors, get_attribute_groups, load_pickle, \
    save_pickle, average_over_folds, load_xlsa17
from misc.log_utils import Logger
from misc.metrics_utils import calc_per_class_centroids, calc_gzsl_metrics
from misc.train_eval_utils import train_model, eval_expandable_model, load_model, create_data_loader, freeze_model, \
    unfreeze_model, get_trainable_params
# from ..DAZLE.train_and_eval import create_attribute_defined_classes_signatures
from zsl_feature_composition.models import DazleCC, DazleComposer, DazleAttributeGrouper, DazleFakeComposer, \
    DazleFakeMixer, DazleFakeComposerUnlimited, DazleFakeComposerUnlimitedScalarAug

from misc.visualize_results import plot_metric_map_from_data


def get_class_attribute_matrix(class_defs_base, class_defs_novel):
    class_differences = np.abs(class_defs_novel[:, np.newaxis, :] - class_defs_base[np.newaxis, :, :])

    class_att_mat = np.argmin(class_differences, axis=1)

    return class_att_mat


def calculate_articles(model, train_loader, attribute_descriptors, all_from_cache):
    model.eval()

    model.reset_cache()
    with torch.no_grad():
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.float().to(model.device)
            batch_labels = batch_labels.long().to(model.device)

            scores = model(batch_imgs)
            batch_package = {'batch_outputs': scores, 'batch_labels': batch_labels}
            model.update_novel_cache(batch_package)

    novel_classes_articles = np.mean(model.cache_samples, axis=1)

    if all_from_cache:
        return novel_classes_articles

    seen_features = train_loader.features
    seen_labels = train_loader.labels
    seen_avg_imgs = calc_per_class_centroids(seen_features, seen_labels, reduce_regions=False)

    shape = seen_avg_imgs.shape
    seen_avg_imgs = np.reshape(seen_avg_imgs, (shape[0], shape[1], -1))

    weights = np.einsum('if,kfr->kir', attribute_descriptors, seen_avg_imgs)
    weights_sftmax = softmax(weights, axis=-1)

    seen_classes_articles = np.einsum('kir,kfr->kif', weights_sftmax, seen_avg_imgs)

    all_articles = np.concatenate([seen_classes_articles, novel_classes_articles], axis=0)
    return all_articles


def calculate_articles2(model, train_loader, attribute_descriptors):
    model.eval()

    model.reset_cache()
    with torch.no_grad():
        for batch_imgs, batch_labels in train_loader:
            batch_imgs = batch_imgs.float().to(model.device)
            batch_labels = batch_labels.long().to(model.device)

            scores = model(batch_imgs)
            batch_package = {'batch_outputs': scores, 'batch_labels': batch_labels}
            model.update_novel_cache(batch_package)

    classes_articles = np.mean(model.cache_samples, axis=1)
    classes_articles = np.transpose(classes_articles, (0, 2, 1))  # make the last channels as regions

    weights = np.einsum('if,kfr->kir', attribute_descriptors, classes_articles)
    weights_sftmax = softmax(weights, axis=-1)

    classes_articles = np.einsum('kir,kfr->kif', weights_sftmax, classes_articles)

    return classes_articles


def calc_class_pairs(base_class_def, novel_class_def, num_pairs, criterion):
    num_base_classes = base_class_def.shape[0]

    if num_pairs <= 0:
        return None

    if criterion == 'closest_to_novel':
        classes_to_approximate = np.random.permutation(novel_class_def.shape[0])[:num_pairs]
        novel_classes_def = novel_class_def[classes_to_approximate]
        new_classes_defs = (base_class_def[:, np.newaxis, :] + base_class_def[np.newaxis, :, :]) * 0.5
        dist = np.sum(np.abs(new_classes_defs[np.newaxis, :, :, :] - np.expand_dims(novel_classes_def, axis=(1, 2))),
                      axis=-1)

        lower_tri_ind = np.tril_indices(dist.shape[1], k=0, m=dist.shape[2])
        mask = np.ones((1, num_base_classes, num_base_classes))
        mask[0, lower_tri_ind[0], lower_tri_ind[1]] = np.inf
        dist *= mask

        pairs = []
        for i in range(num_pairs):
            sorted = np.argsort(dist[i].reshape([1, -1])).squeeze()
            for pair_id in sorted:
                if pair_id not in pairs:
                    pairs.append(pair_id)
                    break

    elif criterion == 'random':
        mat = np.random.uniform(size=(num_base_classes, num_base_classes))
        lower_tri_ind = np.tril_indices(mat.shape[0], k=0, m=mat.shape[1])
        mat[lower_tri_ind] = 0
        mat = np.reshape(mat, newshape=(1, -1)).squeeze()
        pairs = mat.argsort()[-num_pairs:]

    elif criterion == 'farthest_from_base':
        new_classes_defs = (base_class_def[:, np.newaxis, :] + base_class_def[np.newaxis, :, :]) * 0.5
        new_classes_defs = np.reshape(new_classes_defs, newshape=(-1, new_classes_defs.shape[-1]))
        pairs = []
        dist = np.sum(np.abs(base_class_def[np.newaxis, :, :] - new_classes_defs[:, np.newaxis, :]), axis=-1)  # (n,o)

        # an algorithm that chooses from the new class defs one after one and eah one chosen is added as a base class
        for i in range(num_pairs):
            chosen_new_class = np.argmax(np.min(dist, axis=-1))
            pairs.append(chosen_new_class)

            new_dist = np.sum(np.abs(new_classes_defs[chosen_new_class, :] - new_classes_defs[:, :]), axis=-1).squeeze()
            dist = np.concatenate([dist, new_dist[:, np.newaxis]], axis=1)
    else:
        raise ValueError('unsupported criterion')

    assert len(pairs) == num_pairs
    pairs = [[pair_id // num_base_classes, pair_id % num_base_classes] for pair_id in pairs]
    pairs = np.array(pairs)
    return pairs


def visualize_atten():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 16
    test_batch_size = 64
    use_bilinear_gate = True
    sftmax_temp = 1.
    bias = 1
    lam = 0.1
    use_dropout = True

    idx_GPU = 1
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'original with temperature {sftmax_temp}'

    save_folder = f'./exp_grouping/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')

    # data loading
    data_utils.feature_and_labels_cache = {}
    train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size, conv_features=True,
                                        offset=0, shuffle_batch=True).to(device)

    instance_shape = train_loader[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors('bert', dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [train_loader, [test_seen, test_unseen]]

    attribute_groups = None

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    model = DazleAttributeGrouper(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=(lam, 0),
                                  init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                  seen_classes=seen_classes, unseen_classes=unseen_classes,
                                  num_decoders=(1, 0), use_dropout=use_dropout, device=device,
                                  use_bilinear_gate=use_bilinear_gate, attribute_groups=attribute_groups,
                                  cal_unseen=True, bias=bias, dropout_rate=0.05,
                                  attention_sftmax_temperature=sftmax_temp).float()

    model.to(device)

    model, saved_dict = load_model(model_path, model)
    print(saved_dict['stats'])

    res = eval_expandable_model(model, [test_seen, test_unseen])

    z = 4


def main():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 16
    test_batch_size = 64
    num_epochs = 40
    use_CC = True
    bias = 1 if not use_CC else 0
    lam = 0.1 if not use_CC else 0
    sphere_size = 20.
    # bias = 1
    # lam = 0.1
    use_dropout = True

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'with CC sphere size 20'
    print(experiment_desc)
    save_folder = f'./exp_CC/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')

    log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_file.csv')
    global_log = Logger(global_log_path, ['base_network', 'experiment_desc', *metrics_headers],
                        False)

    # data loading
    data_utils.feature_and_labels_cache = {}
    train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size, conv_features=True,
                                        offset=0, shuffle_batch=True).to(device)

    instance_shape = train_loader[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors('bert', dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [train_loader, [test_seen, test_unseen]]

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    model = DazleCC(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=(lam, 0),
                    init_w2v_att=attribute_descriptors,
                    classes_attributes=class_defs, seen_classes=seen_classes, unseen_classes=unseen_classes,
                    num_decoders=(2, 0), use_dropout=use_dropout, device=device, sphere_size=sphere_size,
                    cal_unseen=True,
                    bias=bias, use_CC=use_CC, dropout_rate=0.05).float()

    # model = Dazle(instance_shape[1], w2v_dim, (lam, 0), attribute_descriptors, True,
    #               class_defs, seen_classes, unseen_classes, normalize_V=False,
    #               num_decoders=(2, 0), summarizeing_op='sum', translation_op='no_translation', use_dropout=True,
    #               device=device,
    #               bias=1, cal_unseen=True, norm_instances=False, gt_class_articles=None, backbone=None,
    #               drop_rate=0.05).float()

    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=False)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)

    best_metrics = train_model(model, optimizer=optimizer, scheduler=None, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['experiment_desc'] = experiment_desc
    global_log.log(best_metrics)


def main_composer():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 32
    num_novel_in_batch = 0  # x = num_novel/num_base*batch_size
    test_batch_size = 64
    num_epochs = 25
    bias = 0
    lam = 0.0
    use_dropout = True
    dropout_rate = 0.05
    cache_size = 5
    cache_all_classes = False
    use_offset_cal = True

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'no composer new loss no bias or cal loss but with offset'

    # experiment_desc = 'original with class def normalization'
    print(experiment_desc)
    save_folder = f'./exp_composer/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')

    log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_file.csv')
    global_log = Logger(global_log_path, ['base_network', 'experiment_desc', *metrics_headers], False)

    # data loading
    data_utils.feature_and_labels_cache = {}
    train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size, conv_features=True,
                                        offset=0, shuffle_batch=True).to(device)

    instance_shape = train_loader[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [train_loader, [test_seen, test_unseen]]

    # c2c_mat = get_class_attribute_matrix(test_seen.class_attributes, test_unseen.class_attributes)
    c2c_mat = get_class_attribute_matrix(test_seen.class_attributes,
                                         class_defs if cache_all_classes else test_unseen.class_attributes)

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    lambda_ = (lam, 0)
    model = DazleComposer(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                          init_w2v_att=attribute_descriptors, classes_attributes=class_defs, seen_classes=seen_classes,
                          unseen_classes=unseen_classes, num_decoders=(2, 0), use_dropout=use_dropout, device=device,
                          bias=bias, cal_unseen=True, dropout_rate=dropout_rate, c2c_mat=c2c_mat,
                          num_samples_per_novel=cache_size, num_novel_in_batch=num_novel_in_batch,
                          cache_all_classes=cache_all_classes).float()
    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)

    # model.bias = 0
    # model.lambda_ = (0, 0)
    # model.num_novel_in_batch = 12
    # best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
    #                            loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
    #                            model_path=model_path, device=device, num_epochs=num_epochs,
    #                            custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['experiment_desc'] = experiment_desc
    global_log.log(best_metrics)


def main_2stages():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 32
    num_novel_in_batch = 0  # x = num_novel/num_base*batch_size
    test_batch_size = 64
    num_epochs = 30
    bias = 1
    lam = 0.1
    use_dropout = True
    dropout_rate = 0.05
    cache_size = 30

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'2 stages all articles from cache calc 2 -debug'

    save_folder = f'./exp_composer_2stage/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'dazle_comp_{base_network}_{attribute_type}_{experiment_desc}.pth')

    log_file = os.path.join(save_folder, f'dazle_comp_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_file.csv')
    global_log = Logger(global_log_path, ['base_network', 'experiment_desc', *metrics_headers],
                        False)

    # data loading
    data_utils.feature_and_labels_cache = {}
    train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size, conv_features=True,
                                        offset=0, shuffle_batch=True).to(device)

    instance_shape = train_loader[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [train_loader, [test_seen, test_unseen]]

    # c2c_mat = get_class_attribute_matrix(test_seen.class_attributes, test_unseen.class_attributes)
    c2c_mat = get_class_attribute_matrix(test_seen.class_attributes, class_defs)

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    features_dim = instance_shape[1]
    lambda_ = (lam, 0)
    model = DazleComposer(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                          init_w2v_att=attribute_descriptors, classes_attributes=class_defs, seen_classes=seen_classes,
                          unseen_classes=unseen_classes, num_decoders=(2, 0), use_dropout=use_dropout, device=device,
                          bias=bias, cal_unseen=True, dropout_rate=dropout_rate, c2c_mat=c2c_mat,
                          num_samples_per_novel=cache_size, num_novel_in_batch=num_novel_in_batch).float()
    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=False)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['experiment_desc'] = experiment_desc + 'first_stage'
    global_log.log(best_metrics)

    # second stage
    experiment_desc += ' second_stage'
    model_path = os.path.join(save_folder, f'dazle_comp_{base_network}_{attribute_type}_{experiment_desc}.pth')

    log_file = os.path.join(save_folder, f'dazle_comp_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    attribute_descriptors = load_attribute_descriptors(f'{feature_type}_regions_gmm_means', dataset_name)
    w2v_dim = attribute_descriptors.shape[1]
    # articles = calculate_articles(model, train_loader, attribute_descriptors, all_from_cache=True)
    articles = calculate_articles2(model, train_loader, attribute_descriptors)

    # articles = create_attribute_defined_classes_signatures(train_loader.features, train_loader.labels,
    #                                                        test_unseen.features, test_unseen.labels,
    #                                                        attribute_descriptors)
    model = Dazle(features_dim, w2v_dim, lambda_, attribute_descriptors, True,
                  articles, seen_classes, unseen_classes, normalize_V=False,
                  num_decoders=(2, 0), summarizeing_op='mean', translation_op='s2s_translation', use_dropout=False,
                  device=device, bias=bias, cal_unseen=True, norm_instances=False, gt_class_articles=None,
                  backbone=None,
                  drop_rate=0.05).float()
    model.to(device)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
    scheduler = None
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['experiment_desc'] = experiment_desc
    global_log.log(best_metrics)


def main_grouper():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 16
    test_batch_size = 64
    num_epochs = 30
    use_bilinear_gate = True
    use_grouping = False
    sftmax_temp = 1.
    bias = 1
    lam = 0.1
    use_dropout = True

    idx_GPU = 1
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'original with temperature {sftmax_temp}'

    save_folder = f'./exp_grouping/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')

    log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_file.csv')
    global_log = Logger(global_log_path, ['base_network', 'experiment_desc', *metrics_headers],
                        False)

    # data loading
    data_utils.feature_and_labels_cache = {}
    train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size, conv_features=True,
                                        offset=0, shuffle_batch=True).to(device)

    instance_shape = train_loader[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors('bert', dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [train_loader, [test_seen, test_unseen]]

    attribute_groups = get_attribute_groups(dataset_name) if use_grouping else None

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    model = DazleAttributeGrouper(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=(lam, 0),
                                  init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                  seen_classes=seen_classes, unseen_classes=unseen_classes,
                                  num_decoders=(1, 0), use_dropout=use_dropout, device=device,
                                  use_bilinear_gate=use_bilinear_gate, attribute_groups=attribute_groups,
                                  cal_unseen=True, bias=bias, dropout_rate=0.05,
                                  attention_sftmax_temperature=sftmax_temp).float()

    # model = Dazle(instance_shape[1], w2v_dim, (lam, 0), attribute_descriptors, True,
    #               class_defs, seen_classes, unseen_classes, normalize_V=False,
    #               num_decoders=(2, 0), summarizeing_op='sum', translation_op='no_translation', use_dropout=True,
    #               device=device,
    #               bias=1, cal_unseen=True, norm_instances=False, gt_class_articles=None, backbone=None,
    #               drop_rate=0.05).float()

    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=False)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)

    best_metrics = train_model(model, optimizer=optimizer, scheduler=None, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['experiment_desc'] = experiment_desc
    global_log.log(best_metrics)


def main_fake_input_composer():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 32
    test_batch_size = 64
    num_epochs = 30
    num_epochs_main = 50
    bias = 0
    lam = 0.0
    use_dropout = True
    dropout_rate = 0.05
    use_offset_cal = True
    debug = True

    num_fake_classes = 25
    class_choice_criterion = 'closest_to_novel'  # 'random' 'farthest_from_base'

    criteria = ('closest_to_novel', 'random', 'farthest_from_base')
    num_fake_classes_to_try = [25, 50, 75, 100][::-1]
    my_experiments = [(0, 'x')] + list(itertools.product(num_fake_classes_to_try, criteria))

    input_img_size = 224
    input_img_shape = input_img_size, input_img_size
    use_random_erase = False

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging

    for num_fake_classes, class_choice_criterion in my_experiments[7:]:
        use_dropout = num_fake_classes == 0
        if class_choice_criterion == 'closest_to_novel' and num_fake_classes > 50:
            continue

        experiment_desc = f'{class_choice_criterion} {num_fake_classes} classes - relative in batch'
        experiment_desc = 'debug' if debug else experiment_desc
        print(experiment_desc)

        save_folder = f'./exp_fake_comp_hp_search/'
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, f'composer_{base_network}_{attribute_type}_{experiment_desc}.pth')

        log_file = os.path.join(save_folder, f'log_composer_{base_network}_{attribute_type}_{experiment_desc}.csv')
        metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias']
        exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_file.csv')
        global_log = Logger(global_log_path, ['base_network', 'experiment_desc', *metrics_headers], False)

        # data loading
        num_workers = 0 if debug else 4
        test_transforms = get_testing_transform(input_img_shape)
        test_seen = create_data_loader(dataset_name, 'test_seen_loc', test_batch_size, num_workers,
                                       transform=test_transforms)
        seen_classes = test_seen.dataset.classes

        test_unseen = create_data_loader(dataset_name, 'test_unseen_loc', test_batch_size, num_workers,
                                         transform=test_transforms,
                                         offset=len(seen_classes))
        unseen_classes = test_unseen.dataset.classes

        base_classes_def, novel_classes_def = test_seen.dataset.class_attributes, test_unseen.dataset.class_attributes
        class_defs = np.concatenate([base_classes_def, novel_classes_def], axis=0)

        attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)

        # building class pairs
        class_pairs = calc_class_pairs(base_classes_def, novel_classes_def, num_fake_classes, class_choice_criterion)

        # building train loader
        sampler_options = {'class_pairs': class_pairs, 'batch_size': train_batch_size, 'num_pairs_in_batch': 'relative',
                           'sampler_type': 'Manifold_batch_sampler'}
        if num_fake_classes == 0:
            sampler_options = {'sampler_type': 'balanced_sampler'}

        train_transforms = get_base_transforms(input_img_size, use_random_erase)
        train_loader = create_data_loader(dataset_name, 'all_train_loc', train_batch_size, num_workers,
                                          sampler_opt=sampler_options, transform=train_transforms)
        img_shape = train_loader.dataset[0][0].shape

        data_loaders = [train_loader, [test_seen, test_unseen]]

        # building model
        seed = 214  # 215#
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        w2v_dim = attribute_descriptors.shape[1]
        lambda_ = (lam, 0)
        num_novel_in_batch = train_loader.batch_sampler.num_pairs_in_batch if class_pairs is not None else 0
        model = DazleFakeMixer(features_dim=img_shape, w2v_dim=w2v_dim, lambda_=lambda_,
                               init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                               seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                               use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                               dropout_rate=dropout_rate, num_fake_in_batch=num_novel_in_batch,
                               class_pairs=class_pairs, backbone=feature_type).float()
        model.to(device)

        # training
        def loss_fn(x, y):
            x['batch_label'] = y
            loss_package = model.compute_loss(x)
            return loss_package['loss']

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal)

        # early eval if model exists
        # model_path = './exp_fake_comp_hp_search/composer_densenet_201_T3_bert_x 0 classes - relative in batch.pth'
        # model, saved_dict = load_model(model_path, model)
        # print(saved_dict['stats'])
        # new_results = custom_eval(model, [train_loader, test_unseen])
        # print(new_results)

        # training
        freeze_model(model.backbone)
        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

        optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
        scheduler = None
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)

        unfreeze_model(model.backbone)
        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': 0.00001, 'weight_decay': 0.0001}  # , 'momentum': 0.9}

        optimizer = optim.Adam(params=learnable_params, **optimizer_dflt_params)
        scheduler = None
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
                                   model_path=model_path, device=device, num_epochs=num_epochs_main,
                                   custom_eval_epoch=custom_eval)

        best_metrics['base_network'] = base_network
        best_metrics['experiment_desc'] = experiment_desc
        global_log.log(best_metrics)


def main_fake_input_composer_original_settings():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 32
    num_novel_in_batch = 0  # x = num_novel/num_base*batch_size
    test_batch_size = 64
    num_epochs = 30
    bias = 0
    lam = 0.0
    use_dropout = True
    dropout_rate = 0.05
    cache_size = 5
    cache_all_classes = False
    use_offset_cal = True

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'input fake composer with original settings with offset correction no bias or cal'

    # experiment_desc = 'original with class def normalization'
    print(experiment_desc)
    save_folder = f'./exp_fake_comp/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')

    log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_file.csv')
    global_log = Logger(global_log_path, ['base_network', 'experiment_desc', *metrics_headers], False)

    # data loading
    data_utils.feature_and_labels_cache = {}
    train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size, conv_features=True,
                                        offset=0, shuffle_batch=True).to(device)

    instance_shape = train_loader[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [train_loader, [test_seen, test_unseen]]

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    lambda_ = (lam, 0)
    model = DazleFakeMixer(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                           init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                           seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                           use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                           dropout_rate=dropout_rate, num_fake_in_batch=0,
                           class_pairs=None, backbone=None).float()
    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)

    # model.bias = 0
    # model.lambda_ = (0, 0)
    # model.num_novel_in_batch = 12
    # best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
    #                            loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
    #                            model_path=model_path, device=device, num_epochs=num_epochs,
    #                            custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['experiment_desc'] = experiment_desc
    global_log.log(best_metrics)


def main_fake_deep_composer():
    dataset_name = 'AWA2'
    feature_type = 'resnet_101_L4'  # 'resnet_101_L4', 'densenet_201_T3' , 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'w2v'
    train_batch_size = 50
    test_batch_size = 64
    num_epochs = 30
    bias = 0
    lam = 0.
    use_dropout = False
    dropout_rate = 0.05
    use_offset_cal = True
    norm_v = True
    balance_dataset = True

    idx_GPU = 1
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    vis_drop_rates = [0.5]
    num_fakes_in_bach = [4, 8, 15, 20, 25, 30, 32]
    # vis_drop_rates = [0.]
    # num_fakes_in_bach = [0]
    my_experiments = [(0., 0)]
    my_experiments.extend(list(itertools.product(vis_drop_rates, num_fakes_in_bach)))
    # my_experiments = list(itertools.product(vis_drop_rates, num_fakes_in_bach))

    data_utils.feature_and_labels_cache = {}

    my_experiments = [(0., 0)]
    my_experiments = list(
        itertools.product(['resnet_101_L4'], [False, True], ['bert', 'w2v'],
                          [0.9, 0.], [32]))

    curr = 'resnet_101_L4'

    for exp_num, exp_params in list(enumerate(my_experiments))[:1]:
        vis_drop_rate, num_fake_in_batch = 0., 0

        feature_type, norm_v, attribute_type, momentum, train_batch_size = exp_params
        base_network = feature_type

        if curr != feature_type:
            # mark previous data for gc to collect
            data_utils.feature_and_labels_cache = {}
            train_loader = test_unseen = test_unseen = None
            print(f'freed {curr} features')
            curr = feature_type
            gc.collect()

        # logging
        experiment_desc = f'{num_fake_in_batch} in batch drop rate {vis_drop_rate} with cosine scheduling to 0'
        experiment_desc = f'my code {attribute_type} + norm v {norm_v} + balanced + momentum {momentum} b {train_batch_size}'

        hp_headers = ['norm_v', 'attribute_type', 'momentum']

        print(f'exp number: {exp_num}')
        print(experiment_desc)
        save_folder = f'./deep_dropout_composer_with_cal_bias0/'
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')
        print(model_path)

        log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
        metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
        exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_file_deep_comp.csv')
        global_log = Logger(global_log_path,
                            ['dataset_name', 'base_network', *hp_headers, 'experiment_desc', *metrics_headers],
                            False)

        # data loading
        train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size,
                                            conv_features=True,
                                            offset=0, shuffle_batch=True, balance_dataset=balance_dataset).to(device)

        instance_shape = train_loader[0][0].shape
        shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
        print(shape_str)

        test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                         offset=0).to(device)
        seen_classes = test_seen.classes
        test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size,
                                           conv_features=True,
                                           offset=len(seen_classes)).to(device)

        unseen_classes = test_unseen.classes
        attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
        class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
        data_loaders = [train_loader, [test_seen, test_unseen]]

        # building model
        seed = 214  # 215#
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        w2v_dim = attribute_descriptors.shape[1]
        lambda_ = (lam, 0)
        model = DazleFakeComposerUnlimited(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                                           init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                           seen_classes=seen_classes, unseen_classes=unseen_classes,
                                           num_decoders=(2, 0),
                                           use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                           dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                           vis_drop_rate=vis_drop_rate, normalize_V=norm_v).float()

        # model = DAZLE(instance_shape[1], w2v_dim,
        #               attribute_descriptors, class_defs, None,
        #               seen_classes, unseen_classes,
        #               lambda_[0],
        #               trainable_w2v=True, normalize_V=True, normalize_F=True, is_conservative=True,
        #               uniform_att_1=False, uniform_att_2=False,
        #               prob_prune=0, desired_mass=1, is_conv=False,
        #               is_bias=True, device=device).float()

        model.to(device)

        # training

        def loss_fn(x, y):
            x['batch_label'] = y
            loss_package = model.compute_loss(x)
            return loss_package['loss']

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal,
                                        scores_key='S_pp')

        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': momentum}

        optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
        scheduler = None  # optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)

        # model.bias = 0
        # model.lambda_ = (0, 0)
        # model.num_novel_in_batch = 12
        # best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
        #                            loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
        #                            model_path=model_path, device=device, num_epochs=num_epochs,
        #                            custom_eval_epoch=custom_eval)

        best_metrics['base_network'] = base_network
        best_metrics['dataset_name'] = dataset_name
        best_metrics['experiment_desc'] = experiment_desc
        best_metrics['batch_size'] = train_batch_size

        best_metrics['norm_v'] = norm_v
        best_metrics['attribute_type'] = attribute_type
        best_metrics['momentum'] = momentum

        global_log.log(best_metrics)


def main_fake_deep_composer_scalar_aug(gpu_idx):
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'  # 'resnet_101_L4', 'densenet_201_T3' , 'resnet_101_L3'
    attribute_type = 'bert'
    train_batch_size = 32
    test_batch_size = 128
    num_epochs = 30
    bias = 1
    lam = 0.1
    momentum = 0.9
    use_dropout = True
    dropout_rate = 0.05
    use_offset_cal = False
    norm_v = False
    balance_dataset = True

    idx_GPU = gpu_idx
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    vis_drop_rates = [0.5]
    num_fakes_in_bach = [4, 8, 15, 20, 25, 30, 32, 40, 50]
    # vis_drop_rates = [0.]
    # num_fakes_in_bach = [0]
    my_experiments = [(0., 0, 'binary')]
    my_experiments.extend(list(itertools.product(vis_drop_rates, num_fakes_in_bach, ['binary'])))
    my_experiments.extend(list(itertools.product([0.], num_fakes_in_bach, ['scalar'])))

    data_utils.feature_and_labels_cache = {}

    curr = 'densenet_201_T3'

    for exp_num, exp_params in list(enumerate(my_experiments))[:1]:
        vis_drop_rate, num_fake_in_batch, aug_type = exp_params

        base_network = feature_type

        if curr != feature_type:
            # mark previous data for gc to collect
            data_utils.feature_and_labels_cache = {}
            train_loader = test_unseen = test_unseen = None
            print(f'freed {curr} features')
            curr = feature_type
            gc.collect()

        # logging
        experiment_desc = f'p {vis_drop_rate} - m{num_fake_in_batch} - aug {aug_type} b {train_batch_size}'

        hp_headers = ['vis_drop_rate', 'num_fake', 'aug_type']

        print(f'exp number: {exp_num} / {len(my_experiments)}')
        print(experiment_desc)
        save_folder = f'./deep_dropout_composer_aug/'
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, f'fictitious_{base_network}_{attribute_type}_{experiment_desc}.pth')
        print(model_path)

        log_file = os.path.join(save_folder, f'log_fictitious_{base_network}_{attribute_type}_{experiment_desc}.csv')
        metrics_headers = ['acc_novel', 'acc_seen', 'H', 'acc_zs', 'supervised_acc', 'auc']
        exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_file_deep_comp_aug.csv')
        global_log = Logger(global_log_path,
                            ['dataset_name', 'base_network', 'attribute_type', *hp_headers, 'experiment_desc', *metrics_headers],
                            False)

        # data loading
        train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size,
                                            conv_features=True, offset=0, shuffle_batch=True,
                                            balance_dataset=balance_dataset).to(device)

        instance_shape = train_loader[0][0].shape
        shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
        print(shape_str)

        test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                         offset=0).to(device)
        seen_classes = test_seen.classes
        test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size,
                                           conv_features=True,
                                           offset=len(seen_classes)).to(device)

        unseen_classes = test_unseen.classes
        attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
        class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
        data_loaders = [train_loader, [test_seen, test_unseen]]

        # building model
        seed = 214  # 215#
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        w2v_dim = attribute_descriptors.shape[1]
        lambda_ = (lam, 0)

        model_constructor = DazleFakeComposerUnlimited if aug_type == 'binary' else DazleFakeComposerUnlimitedScalarAug

        # model = model_constructor(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
        #                           init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
        #                           seen_classes=seen_classes, unseen_classes=unseen_classes,
        #                           num_decoders=(2, 0),
        #                           use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
        #                           dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
        #                           vis_drop_rate=vis_drop_rate, normalize_V=norm_v).float()

        model = Dazle(instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_, init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                      trainable_w2v=True,
                 seen_classes=seen_classes, unseen_classes=unseen_classes, normalize_V=False,
                 num_decoders=(2, 0), summarizeing_op='sum', translation_op='no_translation', use_dropout=use_dropout,
                 device=device,
                 bias=bias, cal_unseen=True, norm_instances=False, gt_class_articles=None, backbone=None,
                 drop_rate=0.05, attention_sftmax_temperature=1., normalize_class_defs=True).float()

        model.to(device)

        # training

        def loss_fn(x, y):
            x['batch_label'] = y
            loss_package = model.compute_loss(x)
            return loss_package['loss']

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal,
                                        scores_key='S_pp')

        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': momentum}

        optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
        scheduler = None  # optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)

        # model.bias = 0
        # model.lambda_ = (0, 0)
        # model.num_novel_in_batch = 12
        # best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
        #                            loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
        #                            model_path=model_path, device=device, num_epochs=num_epochs,
        #                            custom_eval_epoch=custom_eval)

        best_metrics['base_network'] = base_network
        best_metrics['dataset_name'] = dataset_name
        best_metrics['experiment_desc'] = experiment_desc
        best_metrics['batch_size'] = train_batch_size

        best_metrics['vis_drop_rate'] = vis_drop_rate
        best_metrics['attribute_type'] = attribute_type
        best_metrics['num_fake'] = num_fake_in_batch
        best_metrics['aug_type'] = aug_type

        global_log.log(best_metrics)


def check_momentum():
    dataset_name = 'CUB'
    feature_type = 'resnet_101_L4'  # 'resnet_101_L4', 'densenet_201_T3' , 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'w2v'
    train_batch_size = 50
    test_batch_size = 64
    num_epochs = 30
    bias = 0
    lam = 0.
    use_dropout = False
    dropout_rate = 0.0
    use_offset_cal = True
    norm_v = True
    balance_dataset = True

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    vis_drop_rates = [0.5]
    num_fakes_in_bach = [4, 8, 15, 20, 25, 30, 32]
    # vis_drop_rates = [0.]
    # num_fakes_in_bach = [0]
    my_experiments = [(0., 0)]
    my_experiments.extend(list(itertools.product(vis_drop_rates, num_fakes_in_bach)))
    # my_experiments = list(itertools.product(vis_drop_rates, num_fakes_in_bach))

    data_utils.feature_and_labels_cache = {}

    my_experiments = [(0., 0)]
    my_experiments = list(
        itertools.product(['resnet_101_L4', 'resnet_101_L4', 'densenet_201_T3'], [False], ['w2v', 'bert'],
                          [0.9], [32], [(0., 0)], [True], [1, 2, 3]))

    norm_class_defs, use_log, scale, offset, log_base = True, False, 1, 0, 0

    curr = 'resnet_101_L4'
    x = 0
    for exp_num, exp_params in list(enumerate(my_experiments)):
        vis_drop_rate, num_fake_in_batch = 0., 0

        feature_type, norm_v, attribute_type, momentum, train_batch_size, hp, balance_dataset, num_decoders = exp_params
        base_network = feature_type

        if not use_log and norm_class_defs and x > 0:
            x += 1
            continue

        vis_drop_rate, num_fake_in_batch = hp

        if curr != feature_type:
            # mark previous data for gc to collect
            data_utils.feature_and_labels_cache = {}
            train_loader = test_unseen = test_unseen = None
            print(f'freed {curr} features')
            curr = feature_type
            gc.collect()

        # logging
        experiment_desc = f'{num_fake_in_batch} in batch drop rate {vis_drop_rate} with cosine scheduling to 0'
        experiment_desc = f'exp_num {exp_num} debug'

        hp_headers = ['norm_v', 'attribute_type', 'momentum', 'keep_rate', 'num_fake', 'balanced', 'num_decoders']

        print(f'exp number: {exp_num}')
        print(experiment_desc)
        save_folder = f'./ablation_study_{dataset_name.lower()}/'
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')
        print(model_path)

        log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
        metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
        exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_file_deep_comp.csv')
        global_log = Logger(global_log_path,
                            ['dataset_name', 'base_network', 'epoch', *hp_headers, 'experiment_desc', *metrics_headers],
                            False)

        # data loading
        train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size,
                                            conv_features=True,
                                            offset=0, shuffle_batch=True, balance_dataset=balance_dataset).to(device)

        instance_shape = train_loader[0][0].shape
        shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
        print(shape_str)

        test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                         offset=0).to(device)
        seen_classes = test_seen.classes
        test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size,
                                           conv_features=True,
                                           offset=len(seen_classes)).to(device)

        unseen_classes = test_unseen.classes
        attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
        class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
        data_loaders = [train_loader, [test_seen, test_unseen]]

        class_defs = class_defs / scale + offset
        if use_log:
            if log_base > 0:
                class_defs = np.log(class_defs) / np.log(log_base)
            else:
                class_defs = np.exp(class_defs)

        # building model
        seed = 214  # 215#
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        w2v_dim = attribute_descriptors.shape[1]
        lambda_ = (lam, 0)
        model = DazleFakeComposerUnlimited(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                                           init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                           seen_classes=seen_classes, unseen_classes=unseen_classes,
                                           num_decoders=(num_decoders, 0),
                                           use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                           dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                           vis_drop_rate=vis_drop_rate, normalize_V=norm_v,
                                           normalize_class_defs=norm_class_defs).float()

        # model = DAZLE(instance_shape[1], w2v_dim,
        #               attribute_descriptors, class_defs, None,
        #               seen_classes, unseen_classes,
        #               lambda_[0],
        #               trainable_w2v=True, normalize_V=True, normalize_F=True, is_conservative=True,
        #               uniform_att_1=False, uniform_att_2=False,
        #               prob_prune=0, desired_mass=1, is_conv=False,
        #               is_bias=True, device=device).float()

        model.to(device)

        # training

        def loss_fn(x, y):
            x['batch_label'] = y
            loss_package = model.compute_loss(x)
            return loss_package['loss']

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal,
                                        scores_key='S_pp')

        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': momentum}

        optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
        scheduler = None  # optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)

        # model.bias = 0
        # model.lambda_ = (0, 0)
        # model.num_novel_in_batch = 12
        # best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
        #                            loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
        #                            model_path=model_path, device=device, num_epochs=num_epochs,
        #                            custom_eval_epoch=custom_eval)

        best_metrics['base_network'] = base_network
        best_metrics['dataset_name'] = dataset_name
        best_metrics['experiment_desc'] = experiment_desc
        best_metrics['batch_size'] = train_batch_size

        best_metrics['norm_v'] = norm_v
        best_metrics['attribute_type'] = attribute_type
        best_metrics['momentum'] = momentum
        best_metrics['keep_rate'] = vis_drop_rate
        best_metrics['num_fake'] = num_fake_in_batch
        best_metrics['balanced'] = balance_dataset

        best_metrics['num_decoders'] = num_decoders

        global_log.log(best_metrics)


def check_optimization():
    dataset_name = 'AWA2'
    feature_type = 'resnet_101_L4'  # 'resnet_101_L4', 'densenet_201_T3' , 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'w2v'
    train_batch_size = 50
    test_batch_size = 64
    num_epochs = 30
    bias = 0
    lam = 0.
    use_dropout = False
    dropout_rate = 0.05
    use_offset_cal = True
    norm_v = True
    balance_dataset = True

    idx_GPU = 1
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    vis_drop_rates = [0.5]
    num_fakes_in_bach = [4, 8, 15, 20, 25, 30, 32]
    # vis_drop_rates = [0.]
    # num_fakes_in_bach = [0]
    my_experiments = [(0., 0)]
    my_experiments.extend(list(itertools.product(vis_drop_rates, num_fakes_in_bach)))
    # my_experiments = list(itertools.product(vis_drop_rates, num_fakes_in_bach))

    data_utils.feature_and_labels_cache = {}

    my_experiments = [(0., 0)]
    my_experiments = list(
        itertools.product(['resnet_101_L4', 'densenet_201_T3'], [False], ['bert'],
                          [0., 0.9], [32], [(0., 0)], [True, False], ['RMSprop', 'Adam', 'SGD'], [True, False],
                          [1e-4, 1e-3, 1e-5]))

    curr = 'resnet_101_L4'

    optimizer_constructor_cache = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam, 'SGD': optim.SGD}

    for exp_num, exp_params in list(enumerate(my_experiments)):
        vis_drop_rate, num_fake_in_batch = 0., 0

        feature_type, norm_v, attribute_type, momentum, train_batch_size, hp, balance_dataset, \
        optimizer_cls, use_cos_sched, init_lr = exp_params
        base_network = feature_type
        optimizer_constructor = optimizer_constructor_cache[optimizer_cls]

        if optimizer_cls == 'Adam' and momentum != 0.:
            continue

        vis_drop_rate, num_fake_in_batch = hp

        if curr != feature_type:
            # mark previous data for gc to collect
            data_utils.feature_and_labels_cache = {}
            train_loader = test_unseen = test_unseen = None
            print(f'freed {curr} features')
            curr = feature_type
            gc.collect()

        # logging
        experiment_desc = f'{num_fake_in_batch} in batch drop rate {vis_drop_rate} with cosine scheduling to 0'
        experiment_desc = f'exp_num {exp_num}'

        hp_headers = ['norm_v', 'attribute_type', 'momentum', 'keep_rate', 'num_fake', 'balanced',
                      'init_lr', 'optimizer', 'use_cosine_sched']

        print(f'exp number: {exp_num}')
        print(experiment_desc)
        save_folder = f'./awa_debug/deep_dropout_composer_check_optimization/'
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')
        print(model_path)

        log_file = os.path.join(save_folder, f'log_apn_{base_network}_{attribute_type}_{experiment_desc}.csv')
        metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc']
        exp_logger = Logger(log_file, ['epoch', 'test_loss', 'lr', *metrics_headers], overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_file_deep_comp.csv')
        global_log = Logger(global_log_path,
                            ['dataset_name', 'base_network', 'epoch', *hp_headers, 'experiment_desc', *metrics_headers],
                            False)

        # data loading
        train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, 'all_train', train_batch_size,
                                            conv_features=True,
                                            offset=0, shuffle_batch=True, balance_dataset=balance_dataset).to(device)

        instance_shape = train_loader[0][0].shape
        shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
        print(shape_str)

        test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                         offset=0).to(device)
        seen_classes = test_seen.classes
        test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size,
                                           conv_features=True,
                                           offset=len(seen_classes)).to(device)

        unseen_classes = test_unseen.classes
        attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
        class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
        data_loaders = [train_loader, [test_seen, test_unseen]]

        # building model
        seed = 214  # 215#
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        w2v_dim = attribute_descriptors.shape[1]
        lambda_ = (lam, 0)
        model = DazleFakeComposerUnlimited(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                                           init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                           seen_classes=seen_classes, unseen_classes=unseen_classes,
                                           num_decoders=(2, 0),
                                           use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                           dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                           vis_drop_rate=vis_drop_rate, normalize_V=norm_v,
                                           normalize_class_defs=True).float()

        # model = DAZLE(instance_shape[1], w2v_dim,
        #               attribute_descriptors, class_defs, None,
        #               seen_classes, unseen_classes,
        #               lambda_[0],
        #               trainable_w2v=True, normalize_V=True, normalize_F=True, is_conservative=True,
        #               uniform_att_1=False, uniform_att_2=False,
        #               prob_prune=0, desired_mass=1, is_conv=False,
        #               is_bias=True, device=device).float()

        model.to(device)

        # training

        def loss_fn(x, y):
            x['batch_label'] = y
            loss_package = model.compute_loss(x)
            return loss_package['loss']

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal,
                                        scores_key='S_pp')

        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': init_lr, 'weight_decay': 0.0001}
        if optimizer_cls != 'Adam':
            optimizer_dflt_params['momentum'] = momentum
        optimizer = optimizer_constructor(params=learnable_params, **optimizer_dflt_params)
        scheduler = None if not use_cos_sched else optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                                        T_max=num_epochs)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)

        # model.bias = 0
        # model.lambda_ = (0, 0)
        # model.num_novel_in_batch = 12
        # best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
        #                            loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
        #                            model_path=model_path, device=device, num_epochs=num_epochs,
        #                            custom_eval_epoch=custom_eval)

        best_metrics['base_network'] = base_network
        best_metrics['dataset_name'] = dataset_name
        best_metrics['experiment_desc'] = experiment_desc
        best_metrics['batch_size'] = train_batch_size

        best_metrics['norm_v'] = norm_v
        best_metrics['attribute_type'] = attribute_type
        best_metrics['momentum'] = momentum
        best_metrics['keep_rate'] = vis_drop_rate
        best_metrics['num_fake'] = num_fake_in_batch
        best_metrics['balanced'] = balance_dataset

        best_metrics['init_lr'] = init_lr
        best_metrics['use_cosine_sched'] = use_cos_sched
        best_metrics['optimizer'] = optimizer_cls

        global_log.log(best_metrics)


def visualize_hp_search():
    hp_search_tag = 'AWA2'
    log_file = f'./exp_fake_deep_comp_hp_search_{hp_search_tag}/global_log_file_deep_comp.csv'

    save_folder = os.path.dirname(log_file)
    metrics2check = ['H', 'acc_zs']
    import pandas as pd
    from misc.visualize_results import plot_metric_map_from_data

    data_frame = pd.read_csv(log_file)
    data = dict()
    for row_idx, row in data_frame.iterrows():
        experiment_tag = row['experiment_desc']
        for metric in metrics2check:

            if experiment_tag == 'original settings':
                num_added_samples = 0
                drop_rate = 0.
            else:
                num_added_samples = int(experiment_tag.split(' in')[0])
                drop_rate = float(experiment_tag.split('rate')[1].split('no')[0])

            if (num_added_samples, drop_rate) not in data:
                data[(num_added_samples, drop_rate)] = dict()

            data[(num_added_samples, drop_rate)][metric] = row[metric]

    for metric in metrics2check:
        plot_metric_map_from_data(data, f'{hp_search_tag} hp search {metric}', save_folder=save_folder,
                                  metric=metric, axis_names=('#added in batch', 'drop_rate'))


def visualize_hp_searchv2():
    hp_search_tag = 'CUB densenet input composition hp search'
    log_file = f'./exp_fake_deep_comp_hp_search_{hp_search_tag}/global_log_file_deep_comp.csv'
    log_file = f'./exp_fake_comp_hp_search/input_comp_global_logfile.csv'

    get_hp1 = lambda exp_desc: exp_desc.split(' ')[0]
    get_hp2 = lambda exp_desc: int(exp_desc.split(' ')[1])
    is_defualt = lambda exp_desc: 'x 0' in exp_desc
    axis_labels = 'criterion', '#classes'

    save_folder = os.path.dirname(log_file)
    metrics2check = ['H', 'acc_zs']
    import pandas as pd

    data_frame = pd.read_csv(log_file)
    data = dict()
    for row_idx, row in data_frame.iterrows():
        experiment_tag = row['experiment_desc']
        for metric in metrics2check:

            if is_defualt(experiment_tag):
                hp1 = '0'
                hp2 = 0.
            else:
                hp1 = get_hp1(experiment_tag)
                hp2 = get_hp2(experiment_tag)

            if (hp1, hp2) not in data:
                data[(hp1, hp2)] = dict()

            data[(hp1, hp2)][metric] = row[metric]

    for metric in metrics2check:
        plot_metric_map_from_data(data, f'{hp_search_tag} hp search {metric}', save_folder=save_folder,
                                  metric=metric,
                                  axis_names=(axis_labels[0], axis_labels[1]))


def plot_validation_pickle(pickle_path):
    # pickle_path = './valid_validations/exp_fake_deep_comp_hp_search_CUB_densenet_201_T3/val_res_CUB_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_CUB_densenet_201_T3/val_res_unlmtd_CUB_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_SUN_densenet_201_T3_32/val_res_unlmtd_SUN_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_AWA2_densenet_201_T3/val_res_unlmtd_AWA2_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_AWA2_densenet_201_T3_64/val_res_unlmtd_AWA2_densenet_201_T3.pkl'
    validation_results = load_pickle(pickle_path)
    axis_labels = 'p - drop percentage', '#added in batch'

    save_folder = os.path.dirname(pickle_path)
    hp_search_tag = pickle_path.split('val_res_')[1].split('.pkl')[0]

    metrics2check = ['H', 'acc_zs', 'auc', 'supervised_acc', 'bias']
    for metric in metrics2check:
        max_hp = max(validation_results, key=lambda x: np.mean(validation_results[x][metric]))
        title = f'{hp_search_tag} hp search {metric} - best hp {max_hp}'
        plot_metric_map_from_data(validation_results, title, save_folder=save_folder,
                                  metric=metric,
                                  axis_names=(axis_labels[0], axis_labels[1]), transpose=True)


def plot_validation_per_fold(pickle_path, fold_id):
    # pickle_path = './valid_validations/exp_fake_deep_comp_hp_search_CUB_densenet_201_T3/val_res_CUB_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_CUB_densenet_201_T3/val_res_unlmtd_CUB_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_SUN_densenet_201_T3_32/val_res_unlmtd_SUN_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_AWA2_densenet_201_T3/val_res_unlmtd_AWA2_densenet_201_T3.pkl'
    # pickle_path = './valid_validations/exp_fake_deep_comp_unlimtd_hp_search_AWA2_densenet_201_T3_64/val_res_unlmtd_AWA2_densenet_201_T3.pkl'
    validation_results = load_pickle(pickle_path)
    axis_labels = 'p - drop percentage', '#added in batch'

    save_folder = os.path.dirname(pickle_path)
    hp_search_tag = pickle_path.split('val_res_')[1].split('.pkl')[0]

    fold_results = {hp: {m: [l[fold_id - 1]] for m, l in d.items()} for hp, d in validation_results.items()}

    metrics2check = ['H', 'acc_zs', 'auc', 'supervised_acc', 'bias']
    for metric in metrics2check:
        max_hp = max(fold_results, key=lambda x: np.mean(fold_results[x][metric]))
        title = f'{hp_search_tag} fold {fold_id} hp search {metric} - best hp {max_hp}'
        plot_metric_map_from_data(fold_results, title, save_folder=save_folder,
                                  metric=metric,
                                  axis_names=(axis_labels[0], axis_labels[1]), transpose=True)


def analyze_output_response():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    test_batch_size = 10
    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05

    num_fake_in_batch = 30  # x = num_novel/num_base*batch_size
    vis_drop_rate = 0.5

    idx_GPU = 0
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    # logging
    experiment_desc = f'{num_fake_in_batch} in batch drop rate {vis_drop_rate} no dropout'

    # experiment_desc = 'original with class def normalization'
    print(experiment_desc)
    save_folder = f'./exp_fake_deep_comp/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'apn_{base_network}_{attribute_type}_{experiment_desc}.pth')

    # data loading
    test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_seen', test_batch_size, conv_features=True,
                                     offset=0).to(device)
    seen_classes = test_seen.classes
    test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, 'test_unseen', test_batch_size, conv_features=True,
                                       offset=len(seen_classes)).to(device)

    unseen_classes = test_unseen.classes
    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
    data_loaders = [test_seen, test_unseen]

    instance_shape = test_seen[0][0].shape
    shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
    print(shape_str)

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    lambda_ = (lam, 0)
    model = DazleFakeComposer(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                              init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                              seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                              use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                              dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                              vis_drop_rate=vis_drop_rate)

    model.to(device)

    load_model(model_path, model, print_stats=True)
    model.eval()

    responses = {'base': [], 'novel': []}
    labels = {'base': [], 'novel': []}
    for data_tag, data_loader in zip(['base', 'novel'], data_loaders):

        for batch_x, batch_y in data_loader:
            with torch.no_grad():
                data_response = model(batch_x)['score_per_class'].detach().cpu().numpy()
                responses[data_tag].append(data_response)
                labels[data_tag].append(batch_y.detach().cpu().numpy())

        responses[data_tag] = np.concatenate(responses[data_tag], axis=0)
        labels[data_tag] = np.concatenate(labels[data_tag], axis=0)

    responses = load_pickle(model_path[:-4] + ' model_responses.pkl')
    x = responses['base']

    base_post_sftmx = softmax(responses['base'], axis=1)
    novel_post_sftmx = softmax(responses['novel'], axis=1)

    all_scores = np.concatenate([base_post_sftmx, novel_post_sftmx], axis=0)
    all_labels = np.concatenate([labels['base'], labels['novel']], axis=0)
    seen_clfrs_scores = all_scores[:, seen_classes]
    unseen_clfrs_scores = all_scores[:, unseen_classes]
    seen_acc, unseen_acc, hm, _, bias = calc_gzsl_metrics(seen_clfrs_scores, unseen_clfrs_scores, all_labels,
                                                          seen_classes, unseen_classes, False, None)
    import plotly.graph_objects as go
    import plotly.offline.offline as py
    from scipy.stats import entropy

    base_max_minus_max = np.max(base_post_sftmx, axis=1) - np.max(base_post_sftmx[:, 150:], axis=1)
    novel_max_minus_max = np.max(novel_post_sftmx, axis=1) - np.max(novel_post_sftmx[:, 150:], axis=1)

    base_entropy = entropy(base_post_sftmx, axis=1)
    novel_entropy = entropy(novel_post_sftmx, axis=1)

    base_max = np.max(base_post_sftmx, axis=1)
    novel_max = np.max(novel_post_sftmx, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=1 - base_max, name='base max', histnorm='probability'))
    fig.add_trace(go.Histogram(x=1 - novel_max, name='novel max', histnorm='probability'))

    # Overlay both histograms
    fig_title = '1-max_softmax'
    fig.update_layout(barmode='overlay', title_text=fig_title)
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    py.plot(fig, filename=f'{fig_title}.html')

    return responses


def finetune_deep_dropout_composer():
    dataset_name = 'AWA2'
    feature_type = 'densenet_201_T3'  # 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 32
    test_batch_size = 64
    num_epochs = 30
    num_main_epochs = 100
    bias = 1
    lam = 0.1
    use_dropout = True
    dropout_rate = 0.05
    use_offset_cal = False
    debug = False
    load_pre_train_chkpnt = False

    use_random_erase = False

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size

    num_fake_in_batch = 0  # x = num_novel/num_base*batch_size
    vis_drop_rate = 0.0

    idx_GPU = 1
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    data_utils.feature_and_labels_cache = {}

    # logging
    experiment_desc = f'end2end {num_fake_in_batch} in batch drop rate {vis_drop_rate} with bias and cal loss'
    # experiment_desc = 'try 2 epochs'
    print(experiment_desc)
    save_folder = f'./exp_fake_deep_comp_end2end/'
    os.makedirs(save_folder, exist_ok=True)
    model_path_pre_train = os.path.join(save_folder,
                                        f'composer_{base_network}_{attribute_type}_{experiment_desc}_pre_train.pth')

    log_file = os.path.join(save_folder,
                            f'log_composer_{dataset_name}_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_exp_fake_deep_comp_end2end.csv')
    global_log = Logger(global_log_path, ['dataset_name', 'base_network', 'experiment_desc', *metrics_headers], False)

    # data loading
    num_workers = 0 if debug else 4
    test_transforms = get_testing_transform(input_img_shape)
    test_seen = create_data_loader(dataset_name, 'test_seen_loc', test_batch_size, num_workers,
                                   transform=test_transforms)
    seen_classes = test_seen.dataset.classes

    test_unseen = create_data_loader(dataset_name, 'test_unseen_loc', test_batch_size, num_workers,
                                     transform=test_transforms,
                                     offset=len(seen_classes))
    unseen_classes = test_unseen.dataset.classes

    base_classes_def, novel_classes_def = test_seen.dataset.class_attributes, test_unseen.dataset.class_attributes
    class_defs = np.concatenate([base_classes_def, novel_classes_def], axis=0)

    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)

    # building train loader
    sampler_options = {'sampler_type': 'batched_balanced_sampler'}

    train_transforms = get_base_transforms(input_img_size, use_random_erase)
    train_loader = create_data_loader(dataset_name, 'all_train_loc', train_batch_size, num_workers,
                                      sampler_opt=sampler_options, transform=train_transforms)
    img_shape = train_loader.dataset[0][0].shape

    data_loaders = [train_loader, [test_seen, test_unseen]]

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    lambda_ = (lam, 0)
    model = DazleFakeComposer(features_dim=img_shape, w2v_dim=w2v_dim, lambda_=lambda_,
                              init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                              seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                              use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                              dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                              vis_drop_rate=vis_drop_rate, backbone=feature_type)

    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal)

    freeze_model(model.backbone)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}

    optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    if not load_pre_train_chkpnt:
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path_pre_train, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)
        best_metrics['base_network'] = base_network
        best_metrics['dataset_name'] = dataset_name
        best_metrics['experiment_desc'] = experiment_desc
        global_log.log(best_metrics)

    second_desc = 'AdamW lr 1e-5 wdecay 5e-4 no scheduledr 100 epochs'
    model_path_main_train = os.path.join(save_folder,
                                         f'composer_{base_network}_{attribute_type}_{experiment_desc}_{second_desc}.pth')
    unfreeze_model(model.backbone)
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params=learnable_params, lr=1e-5, weight_decay=1e-4)
    scheduler = None
    if load_pre_train_chkpnt:
        saved_dict = load_model(model_path_main_train, model)
        optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
        num_main_epochs -= saved_dict['stats']['epoch']

    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
                               model_path=model_path_main_train, device=device, num_epochs=num_main_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['dataset_name'] = dataset_name
    best_metrics['experiment_desc'] = f'{experiment_desc}_{second_desc}'
    global_log.log(best_metrics)


def validate_deep_model(model_constructor, lambdas, folds, dataset_name, data_loader_options,
                        num_warmup_epochs, num_epochs, run_tag, log_file, exp_metrics, optimizers_options, debug=False,
                        save_folder='./'):
    train_batch_size = data_loader_options['train_batch_size']
    test_batch_size = data_loader_options['test_batch_size']

    train_transforms = data_loader_options['train_transforms']
    testing_transforms = data_loader_options['testing_transforms']

    train_sampler_opt = data_loader_options['train_sampler_opt']

    warmup_optimizer_constructor = optimizers_options['warmup_optimizer_constructor']
    main_optimizer_constructor = optimizers_options['main_optimizer_constructor']
    main_scheduler_constructor = optimizers_options['scheduler_constructor']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=True)

    save_folder = f'{save_folder}/validations/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, 'temp.pth')

    lambdas1, lambdas2 = lambdas
    exp_desc = run_tag
    exp_headers = ['epoch', 'test_loss', *exp_metrics]
    val_log = Logger(os.path.join(save_folder, f'val_log {exp_desc}.csv'),
                     ['experiment_desc', *exp_headers], overwrite=False)
    valres_pkl_path = os.path.join(save_folder, f'val_res {exp_desc}.pkl')
    csv_files = []

    model = None
    for fold_id in folds:
        print('-' * 60)
        print(f'fold {fold_id}')

        num_workers = 0 if debug else 4

        train_subset_loader = create_data_loader(dataset_name, f'train_gzsl_loc{fold_id}', train_batch_size,
                                                 num_workers, transform=train_transforms, sampler_opt=train_sampler_opt)

        test_seen_subset_loader = create_data_loader(dataset_name, f'val_seen_loc{fold_id}', test_batch_size,
                                                     num_workers, transform=testing_transforms, mix_type=None)

        seen_classes = train_subset_loader.dataset.classes
        num_seen_classes = len(np.unique(seen_classes))

        test_novel_subset_loader = create_data_loader(dataset_name, f'val_unseen_loc{fold_id}', test_batch_size,
                                                      num_workers, transform=testing_transforms, mix_type=None,
                                                      offset=num_seen_classes)
        unseen_classes = test_novel_subset_loader.dataset.classes

        data_loaders = [train_subset_loader, [test_seen_subset_loader, test_novel_subset_loader]]
        seen_classes_attributes = train_subset_loader.dataset.class_attributes
        novel_classes_attributes = test_novel_subset_loader.dataset.class_attributes
        class_defs = np.concatenate([seen_classes_attributes, novel_classes_attributes], axis=0)
        img_shape = train_subset_loader.dataset[0][0].shape

        all_experiments = itertools.product(lambdas1, lambdas2)
        for lambda_1, lambda_2 in all_experiments:

            run_tag = exp_desc + f' lamb1 {lambda_1} lamb2 {lambda_2} fold {fold_id}'.replace(',', '-')
            fold_log = os.path.join(save_folder, log_file + f' {run_tag}.csv')
            exp_logger = Logger(fold_log, headers=exp_headers, overwrite=False)
            csv_files.append(fold_log)

            exp_key = (lambda_1, lambda_2)
            if is_exp_done(valres_pkl_path, exp_key, fold_id, 'H'):
                print('skipping')
                continue

            seed = 214  # 215#
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            exp_lambdas = (lambda_1, lambda_2, 0)
            model = model_constructor(features_dim=img_shape, seen_classes=seen_classes, unseen_classes=unseen_classes,
                                      classes_attributes=class_defs)  # type: DazleFakeComposer

            model.float()

            def loss_fn(x, y):
                x['batch_label'] = y
                loss_package = model.compute_loss(x)
                return loss_package['loss']

            # warm-up: train new layers only
            freeze_model(model.backbone)
            learnable_params = get_trainable_params(model)
            model.to(model.device)
            # optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}
            # optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)

            optimizer = warmup_optimizer_constructor(params=learnable_params)
            best_metrics = {'main_metric': 0}
            best_metrics = train_model(model, optimizer=optimizer, scheduler=None, dataset_loaders=data_loaders,
                                       loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger,
                                       best_metrics=best_metrics,
                                       model_path=model_path, device=model.device, num_epochs=num_warmup_epochs,
                                       custom_eval_epoch=custom_eval)

            # main train
            unfreeze_model(model.backbone)
            learnable_params = get_trainable_params(model)

            model.to(model.device)
            optimizer = main_optimizer_constructor(params=learnable_params)
            scheduler = None if main_scheduler_constructor is None else main_scheduler_constructor(optimizer=optimizer)

            best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler,
                                       dataset_loaders=data_loaders,
                                       loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger,
                                       best_metrics=best_metrics,
                                       model_path=model_path, device=model.device, num_epochs=num_epochs,
                                       custom_eval_epoch=custom_eval)

            best_metrics['experiment_desc'] = run_tag
            val_log.log(best_metrics)

            log_val_res_pkl(valres_pkl_path, exp_key, best_metrics, exp_headers)

            print('-' * 60)
            print(f'done with fold {fold_id} and hp {exp_key}')
            print('-' * 60)

            del model
            torch.cuda.empty_cache()

    val_res = load_pickle(valres_pkl_path)
    average_over_folds(csv_files)
    best_hyper = max(val_res, key=lambda k: np.mean(val_res.get(k)['H']))
    return best_hyper, np.mean(val_res[best_hyper]['H']), val_res


def validate_deep_dropout_composer(dataset_name, feature_type, best_hp, train_batch_size=32, gpu_idx=0):
    # dataset_name = 'AWA2'
    gc.collect()
    torch.cuda.empty_cache()
    # feature_type = 'resnet_101_L4'  # 'densenet_201_T3' , 'resnet_101_L4', 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'bert'
    # train_batch_size = 32
    test_batch_size = 128
    num_epochs = 30
    num_main_epochs = 50
    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05
    use_offset_cal = False
    debug = False

    use_random_erase = False
    best_bias = None
    do_validation = True

    use_unlimited = True

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size

    # put your hyper parameters here
    num_fake_in_batch = 8  # for CUB:= 70, for AWA2:= 8
    vis_drop_rate = 0.5  # for CUB:= 0.5, for AWA2:= 0.5
    vis_drop_rate, num_fake_in_batch = best_hp

    pre_lr = 1e-4
    main_lr = 1e-5

    momentum = 0.9
    if dataset_name == 'AWA2':
        momentum = 0.
        print('no momentum')

    idx_GPU = gpu_idx
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    data_utils.feature_and_labels_cache = {}
    num_workers = 0 if debug else 4
    sampler_options = {'sampler_type': 'batched_balanced_sampler'}
    train_transforms = get_base_transforms(input_img_size, use_random_erase)
    test_transforms = get_testing_transform(input_img_shape)

    data_loader_options = {'train_batch_size': train_batch_size, 'test_batch_size': test_batch_size,
                           'train_transforms': train_transforms, 'testing_transforms': test_transforms,
                           'train_sampler_opt': sampler_options}

    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    w2v_dim = attribute_descriptors.shape[1]

    model_constructor = functools.partial(DazleFakeComposerUnlimited, w2v_dim=w2v_dim, lambda_=(0, 0),
                                          init_w2v_att=attribute_descriptors, num_decoders=(2, 0),
                                          use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                          dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                          vis_drop_rate=vis_drop_rate, backbone=feature_type)

    experiment_desc = f'{dataset_name} b {train_batch_size} m {num_fake_in_batch} - p {vis_drop_rate}'
    # experiment_desc = f'debug'

    optimizer_constructor = functools.partial(optim.RMSprop, lr=pre_lr, weight_decay=1e-4, momentum=momentum)
    main_optimizer_constructor = functools.partial(optim.AdamW, lr=main_lr, weight_decay=1e-4)
    # scheduler_constructor = functools.partial(optim.lr_scheduler.CosineAnnealingLR, T_max=num_main_epochs)
    scheduler_constructor = None

    optimizer_options = {'warmup_optimizer_constructor': optimizer_constructor,
                         'main_optimizer_constructor': main_optimizer_constructor,
                         'scheduler_constructor': scheduler_constructor}

    # logging
    # experiment_desc = 'try 2 epochs'
    print(experiment_desc)
    save_folder = f'./final_results/reproduce/{dataset_name.lower()}/exp_unlmtd_fake_deep_comp_validate_end2end_{dataset_name}_{feature_type}_{experiment_desc}/'
    # save_folder = f'./tmp'
    os.makedirs(save_folder, exist_ok=True)
    model_path_pre_train = os.path.join(save_folder,
                                        f'composer_{dataset_name}_{base_network}_{attribute_type}_{experiment_desc}_pre_train.pth')

    log_file = os.path.join(save_folder, f'log_composer_{base_network}_{attribute_type}_{experiment_desc}.csv')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    global_log_path = os.path.join(save_folder, f'global_log_exp_fake_deep_comp_end2end.csv')
    global_log = Logger(global_log_path, ['dataset_name', 'base_network', 'experiment_desc', *metrics_headers], False)

    # validation
    if do_validation:
        val_results = validate_deep_model(model_constructor, [[num_fake_in_batch], [vis_drop_rate]], [1, 2, 3],
                                          dataset_name,
                                          data_loader_options, num_warmup_epochs=num_epochs, num_epochs=num_main_epochs,
                                          run_tag=experiment_desc,
                                          log_file=f'log_composer_{base_network}_{attribute_type}_',
                                          exp_metrics=metrics_headers, optimizers_options=optimizer_options,
                                          debug=debug, save_folder=save_folder)

        best_hyper, best_H_val, val_res = val_results
        best_bias = np.mean(val_res[best_hyper]['bias'])
        print(f'best bias is: {best_bias}')
        torch.cuda.empty_cache()
    # data loading

    test_seen = create_data_loader(dataset_name, 'test_seen_loc', test_batch_size, num_workers,
                                   transform=test_transforms)
    seen_classes = test_seen.dataset.classes

    test_unseen = create_data_loader(dataset_name, 'test_unseen_loc', test_batch_size, num_workers,
                                     transform=test_transforms,
                                     offset=len(seen_classes))
    unseen_classes = test_unseen.dataset.classes

    base_classes_def, novel_classes_def = test_seen.dataset.class_attributes, test_unseen.dataset.class_attributes
    class_defs = np.concatenate([base_classes_def, novel_classes_def], axis=0)

    # building train loader
    train_loader = create_data_loader(dataset_name, 'all_train_loc', train_batch_size, num_workers,
                                      sampler_opt=sampler_options, transform=train_transforms)
    img_shape = train_loader.dataset[0][0].shape

    data_loaders = [train_loader, [test_seen, test_unseen]]

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    lambda_ = (lam, 0)
    model_constructor = DazleFakeComposer
    if use_unlimited:
        model_constructor = DazleFakeComposerUnlimited
    model = model_constructor(features_dim=img_shape, w2v_dim=w2v_dim, lambda_=lambda_,
                              init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                              seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                              use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                              dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                              vis_drop_rate=vis_drop_rate, backbone=feature_type)

    model.to(device)

    # training

    def loss_fn(x, y):
        x['batch_label'] = y
        loss_package = model.compute_loss(x)
        return loss_package['loss']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=True, fixed_bias=best_bias)

    freeze_model(model.backbone)

    learnable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': 0.9}
    # optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
    optimizer = optimizer_constructor(params=learnable_params)
    scheduler = None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                               model_path=model_path_pre_train, device=device, num_epochs=num_epochs,
                               custom_eval_epoch=custom_eval)
    best_metrics['base_network'] = base_network
    best_metrics['dataset_name'] = dataset_name
    best_metrics['experiment_desc'] = experiment_desc
    global_log.log(best_metrics)

    second_desc = 'finetuned'
    model_path_main_train = os.path.join(save_folder,
                                         f'composer_{base_network}_{attribute_type}_{experiment_desc}_{second_desc}.pth')
    unfreeze_model(model.backbone)
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.AdamW(params=learnable_params, lr=1e-5, weight_decay=1e-4)
    optimizer = main_optimizer_constructor(params=learnable_params)
    scheduler = scheduler_constructor(optimizer=optimizer) if scheduler_constructor is not None else None
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 20)
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
                               model_path=model_path_main_train, device=device, num_epochs=num_main_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['dataset_name'] = dataset_name
    best_metrics['experiment_desc'] = f'{experiment_desc}_{second_desc}'
    global_log.log(best_metrics)


def load_and_eval_deep_fake_composer():
    dataset_name = 'CUB'
    feature_type = 'densenet_201_T3'
    base_network = feature_type
    attribute_type = 'bert'
    test_batch_size = 64
    use_offset_cal = False
    debug = False
    use_dropout = False
    dropout_rate = 0.05
    bias = 0
    lam = 0.0

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size

    num_fake_in_batch = 70  # x = num_novel/num_base*batch_size
    vis_drop_rate = 0.25

    idx_GPU = 1
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    data_utils.feature_and_labels_cache = {}

    # data loading
    num_workers = 0 if debug else 4
    test_transforms = get_testing_transform(input_img_shape)
    test_seen = create_data_loader(dataset_name, 'test_seen_loc', test_batch_size, num_workers,
                                   transform=test_transforms)
    seen_classes = test_seen.dataset.classes

    test_unseen = create_data_loader(dataset_name, 'test_unseen_loc', test_batch_size, num_workers,
                                     transform=test_transforms,
                                     offset=len(seen_classes))
    unseen_classes = test_unseen.dataset.classes

    base_classes_def, novel_classes_def = test_seen.dataset.class_attributes, test_unseen.dataset.class_attributes
    class_defs = np.concatenate([base_classes_def, novel_classes_def], axis=0)

    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    img_shape = test_seen.dataset[0][0].shape

    data_loaders = [test_seen, test_unseen]

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    w2v_dim = attribute_descriptors.shape[1]
    lambda_ = (lam, 0)
    model = DazleFakeComposerUnlimited(features_dim=img_shape, w2v_dim=w2v_dim, lambda_=lambda_,
                                       init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                       seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                                       use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                       dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                       vis_drop_rate=vis_drop_rate, backbone=feature_type)

    model.to(device)

    model_path_main_train = './final_results/awa/exp_unlmtd_fake_deep_comp_validate_end2end_AWA2_resnet_101_L4_AWA2 b 32 m 8 - p 0.25/' \
                            'mode2check.pth'

    model_path_main_train = './final_results/cub/exp_unlmtd_fake_deep_comp_validate_end2end_CUB_densenet_201_T3/' \
                            'model2check.pth'

    load_model(model_path_main_train, model, print_stats=True)

    results = eval_expandable_model(model, data_loaders, use_post_bias_correction=True)


def do_valid_hp_search(dataset_name, feature_type, balance_dataset=True, gpu_idx=0):
    # dataset_name = 'SUN'
    base_network = feature_type
    attribute_type = 'bert'
    train_batch_size = 32
    test_batch_size = 128
    num_epochs = 30
    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05
    use_offset_cal = True

    momentum = 0.9

    if dataset_name == 'AWA2':
        momentum = 0.
        print('no momentum')

    idx_GPU = gpu_idx
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    data_utils.feature_and_labels_cache = {}
    # global logging
    save_folder = f'./valid_validations/reproduce/{dataset_name.lower()}/exp_fake_deep_comp_unlimtd_hp_search_{dataset_name}_{feature_type}_{train_batch_size}_{momentum}/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, f'temp.pth')
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']
    global_log_path = os.path.join(save_folder, f'global_log_file_deep_comp_unlmtd.csv')
    val_res_pkl_path = os.path.join(save_folder, f'val_res_unlmtd_{dataset_name}_{feature_type}.pkl')

    global_log = Logger(global_log_path, ['dataset_name', 'base_network', 'epoch', 'experiment_desc', *metrics_headers],
                        False)

    # experiments parameters
    vis_drop_rates = [0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9]
    num_fakes_in_bach = [4, 8, 15, 20, 25, 30, 40, 50, 60, 70, 80]
    folds = [1, 2, 3]
    my_experiments = [(0., 0)]

    my_experiments.extend(list(itertools.product(vis_drop_rates, num_fakes_in_bach)))
    my_experiments = list(itertools.product(my_experiments, folds))

    # my_experiments = [((0.5, 40), 1)] for debugging

    for exp_num, exp_params in list(enumerate(my_experiments)):

        exp_key, fold_id = exp_params
        vis_drop_rate, num_fake_in_batch = exp_key
        experiment_desc = f'{num_fake_in_batch} in batch drop rate {vis_drop_rate} fold {fold_id}'
        print(f'exp number: {exp_num}')
        print(experiment_desc)

        if is_exp_done(val_res_pkl_path, exp_key, fold_id, 'H'):
            print('skipping')
            continue

        # data loading
        train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, f'train_gzsl', train_batch_size,
                                            conv_features=True, offset=0, shuffle_batch=True, split_id=fold_id,
                                            balance_dataset=balance_dataset).to(
            device)

        instance_shape = train_loader[0][0].shape
        shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
        print(shape_str)

        test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, f'val_seen', test_batch_size,
                                         conv_features=True, offset=0, split_id=fold_id).to(device)
        seen_classes = test_seen.classes
        test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, f'val_unseen', test_batch_size,
                                           conv_features=True, offset=len(seen_classes), split_id=fold_id).to(device)

        unseen_classes = test_unseen.classes
        attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
        class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
        data_loaders = [train_loader, [test_seen, test_unseen]]

        # logging
        log_file = os.path.join(save_folder,
                                f'log_deep_composer_{dataset_name}_{base_network}_{attribute_type}_{experiment_desc}.csv')
        exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

        # building model
        seed = 214  # 215#
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        w2v_dim = attribute_descriptors.shape[1]
        lambda_ = (lam, 0)
        model = DazleFakeComposerUnlimited(features_dim=instance_shape[1], w2v_dim=w2v_dim, lambda_=lambda_,
                                           init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                           seen_classes=seen_classes, unseen_classes=unseen_classes,
                                           num_decoders=(2, 0),
                                           use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                           dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                           vis_drop_rate=vis_drop_rate)

        model.to(device)

        # training

        def loss_fn(x, y):
            x['batch_label'] = y
            loss_package = model.compute_loss(x)
            return loss_package['loss']

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal, calc_auc=True)

        learnable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_dflt_params = {'lr': 0.0001, 'weight_decay': 0.0001, 'momentum': momentum}

        optimizer = optim.RMSprop(params=learnable_params, **optimizer_dflt_params)
        scheduler = None
        best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                                   loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=None,
                                   model_path=model_path, device=device, num_epochs=num_epochs,
                                   custom_eval_epoch=custom_eval)

        best_metrics['dataset_name'] = dataset_name
        best_metrics['base_network'] = base_network
        best_metrics['experiment_desc'] = experiment_desc
        global_log.log(best_metrics)

        log_val_res_pkl(val_res_pkl_path, exp_key, exp_results=best_metrics, headers=metrics_headers)

    val_res = load_pickle(val_res_pkl_path)
    best_hp = max(val_res, key=lambda x: np.mean(val_res[x]['H']))
    return val_res_pkl_path, val_res, best_hp


def log_val_res_pkl(pkl_path, exp_key, exp_results, headers):
    val_res = load_pickle(pkl_path)
    if val_res is None:
        val_res = dict()
    if exp_key in val_res:
        for h in headers:
            try:
                val_res[exp_key][h].append(exp_results[h])
            except:
                continue
    else:
        val_res[exp_key] = dict()
        for h in headers:
            try:
                val_res[exp_key][h] = [exp_results[h]]
            except:
                continue

    save_pickle(pkl_path, val_res)


def is_exp_done(pkl_path, exp_key, fold_id, metric):
    val_res = load_pickle(pkl_path)
    if val_res is None or val_res.get(exp_key, None) is None:
        return False
    return len(val_res[exp_key][metric]) >= fold_id


def check_data_split_is_valid():
    for dataset in ['CUB', 'SUN', 'AWA2']:
        for fold in range(1, 4):
            tr_imgs, tr_labels, _ = load_xlsa17(dataset, f'train_gzsl_loc{fold}', return_image_paths=True)
            val_seen_imgs, val_seen_labels, _ = load_xlsa17(dataset, f'val_seen_loc{fold}', return_image_paths=True)
            val_unseen_imgs, val_unseen_labels, _ = load_xlsa17(dataset, f'val_unseen_loc{fold}',
                                                                return_image_paths=True)

            tr_classes = np.unique(tr_labels)
            val_seen_classes = np.unique(val_seen_labels)
            val_unseen_classes = np.unique(val_unseen_labels)

            assert np.setdiff1d(tr_classes, val_seen_classes).size == 0
            assert np.intersect1d(tr_classes, val_unseen_classes).size == 0

            assert np.intersect1d(val_seen_imgs, tr_imgs).size == 0
            assert np.intersect1d(tr_imgs, val_unseen_imgs).size == 0

            print(f'fold {fold} in {dataset} checks out')

        tr_imgs, tr_labels, _ = load_xlsa17(dataset, f'all_train_loc', return_image_paths=True)
        val_seen_imgs, val_seen_labels, _ = load_xlsa17(dataset, f'test_seen_loc', return_image_paths=True)
        val_unseen_imgs, val_unseen_labels, _ = load_xlsa17(dataset, f'test_unseen_loc', return_image_paths=True)

        tr_classes = np.unique(tr_labels)
        val_seen_classes = np.unique(val_seen_labels)
        val_unseen_classes = np.unique(val_unseen_labels)

        assert np.setdiff1d(tr_classes, val_seen_classes).size == 0
        assert np.intersect1d(tr_classes, val_unseen_classes).size == 0

        assert np.intersect1d(val_seen_imgs, tr_imgs).size == 0
        assert np.intersect1d(tr_imgs, val_unseen_imgs).size == 0

        print(f'main split in {dataset} checks out')


if __name__ == '__main__':
    # main()
    # main_composer()
    # main_fake_input_composer_original_settings()
    # main_fake_deep_composer()
    # main_fake_deep_composer()
    # visualize_hp_search()
    # main_2stages()
    # main_grouper()
    # visualize_atten()
    # finetune_deep_dropout_composer()

    # validate_deep_dropout_composer()
    # analyze_output_response()

    # load_and_eval_deep_fake_composer()

    # visualize_hp_searchv2()
    # plot_validation_pickle(None)

    # check_data_split_is_valid()
    # check_momentum()
    # import misc.data_utils as du
    #
    # du.feature_and_labels_cache = {}
    # check_optimization()

    # main_fake_deep_composer()

    # load_and_eval_deep_fake_composer()
    # pkl_path = glob('/home/mohammed/Desktop/research_ZSL/zsl_feature_composition/valid_validations/awa2/*L3*/*.pkl')[0]
    # # pkl_path = '/home/mohammed/Desktop/research_ZSL/zsl_feature_composition/valid_validations/awa2/exp_fake_deep_comp_unlimtd_hp_search_AWA2_resnet_101_L4_32_0.0/' \
    # #            'val_res_unlmtd_AWA2_resnet_101_L4.pkl'
    # plot_validation_pickle(pkl_path)
    # for fold in range(1, 4):
    #     plot_validation_per_fold(pkl_path, fold)

    # ablation_study()

    main_fake_deep_composer_scalar_aug(gpu_idx=0)

    # base_model = False
    # dataset = 'SUN'
    # features = 'densenet_201_T3'
    # if base_model:
    #     validate_deep_dropout_composer(dataset_name=dataset, feature_type=features, best_hp=(0., 0),
    #                                    train_batch_size=32, gpu_idx=1)
    #
    # else:
    #     pkl_path, val_res, best_hp = do_valid_hp_search(dataset, feature_type=features, balance_dataset=True, gpu_idx=0)
    #
    #     plot_validation_pickle(pkl_path)
    #     for fold in range(1, 4):
    #         plot_validation_per_fold(pkl_path, fold)
    #     # # load_and_eval_deep_fake_composer()
    #     validate_deep_dropout_composer(dataset_name=dataset, feature_type=features, best_hp=best_hp,
    #                                    train_batch_size=32, gpu_idx=0)

    # best_hp = (0.5, 15)
    # validate_deep_dropout_composer(dataset_name=dataset, feature_type=features, best_hp=best_hp, train_batch_size=32,
    #                                gpu_idx=1)

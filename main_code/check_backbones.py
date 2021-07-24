import argparse
import functools
import gc
import os
import numpy as np
import torch
from torch import optim

from misc import data_utils
from misc.data_loader import ZSLDatasetEmbeddings
from misc.data_utils import load_attribute_descriptors, load_pickle, average_over_folds
from misc.log_utils import Logger
from misc.train_eval_utils import train_model, eval_expandable_model, get_trainable_params
from main_code.models import DazleFakeComposerUnlimited
from main_code.reproduce import is_exp_done, log_val_res_pkl


def validate_backbones(model_constructor, backbones, folds, dataset_name, data_loader_options,
                       num_epochs, run_tag, log_file, exp_metrics, optimizers_options, save_folder='./'):
    train_batch_size = data_loader_options['train_batch_size']
    test_batch_size = data_loader_options['test_batch_size']

    balance_dataset = data_loader_options['balance_dataset']

    optimizer_constructor = optimizers_options['optimizer_constructor']

    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=True, calc_auc=True)

    save_folder = f'{save_folder}/validations/'
    os.makedirs(save_folder, exist_ok=True)
    model_path = os.path.join(save_folder, 'temp.pth')

    exp_desc = run_tag
    exp_headers = ['epoch', 'test_loss', *exp_metrics]
    val_log = Logger(os.path.join(save_folder, f'val_log {exp_desc}.csv'),
                     ['dataset_name', 'backbone', 'experiment_desc', *exp_headers], overwrite=False)

    valres_pkl_path = os.path.join(save_folder, f'val_res {exp_desc}.pkl')
    csv_files = []

    model = None
    for backbone in backbones:

        data_utils.feature_and_labels_cache = {}
        gc.collect()
        torch.cuda.empty_cache()

        for fold_id in folds:
            print('-' * 60)
            print(f'fold {fold_id}')

            run_tag = exp_desc + f' backbone {backbone} fold {fold_id}'.replace(',', '-')
            fold_log = os.path.join(save_folder, log_file + f' {run_tag}.csv')
            exp_logger = Logger(fold_log, headers=exp_headers, overwrite=False)
            csv_files.append(fold_log)

            exp_key = backbone
            if is_exp_done(valres_pkl_path, exp_key, fold_id, 'H'):
                print('skipping')
                continue

            train_loader = ZSLDatasetEmbeddings(dataset_name, backbone, f'train_gzsl', train_batch_size,
                                                split_id=fold_id, conv_features=True, offset=0,
                                                shuffle_batch=True, balance_dataset=balance_dataset)

            test_seen = ZSLDatasetEmbeddings(dataset_name, backbone, 'val_seen', test_batch_size, split_id=fold_id,
                                             conv_features=True, offset=0)
            seen_classes = test_seen.classes
            test_unseen = ZSLDatasetEmbeddings(dataset_name, backbone, 'val_unseen', test_batch_size, split_id=fold_id,
                                               conv_features=True, offset=len(seen_classes))

            unseen_classes = test_unseen.classes
            class_defs = np.concatenate([test_seen.class_attributes, test_unseen.class_attributes], axis=0)
            data_loaders = [train_loader, [test_seen, test_unseen]]

            dim_f = train_loader[0][0].shape[1]

            seed = 214  # 215#
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            model = model_constructor(features_dim=dim_f, seen_classes=seen_classes, unseen_classes=unseen_classes,
                                      classes_attributes=class_defs)  # type: Dazle

            model.float()

            def loss_fn(x, y):
                x['batch_label'] = y
                loss_package = model.compute_loss(x)
                return loss_package['loss']

            # warm-up: train new layers only
            learnable_params = get_trainable_params(model)
            model.to(model.device)
            train_loader.to(model.device)
            test_seen.to(model.device)
            test_unseen.to(model.device)

            optimizer = optimizer_constructor(params=learnable_params)
            best_metrics = {'main_metric': 0}
            best_metrics = train_model(model, optimizer=optimizer, scheduler=None, dataset_loaders=data_loaders,
                                       loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger,
                                       best_metrics=best_metrics, model_path=model_path, device=model.device,
                                       num_epochs=num_epochs, custom_eval_epoch=custom_eval)

            best_metrics['dataset_name'] = dataset_name
            best_metrics['backbone'] = backbone

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


def check_backbones(dataset_name, backbones, train_batch_size=32, gpu_idx=0):
    gc.collect()
    torch.cuda.empty_cache()
    attribute_type = 'bert'
    test_batch_size = 128
    bias = 0
    use_dropout = False
    dropout_rate = 0.05

    vis_drop_rate, num_fake_in_batch = 0., 0

    pre_lr = 1e-4

    momentum = 0.9
    if dataset_name == 'AWA2':
        momentum = 0.
        print('no momentum')

    idx_GPU = gpu_idx
    device = torch.device("cuda:{}".format(idx_GPU) if torch.cuda.is_available() else "cpu")

    data_utils.feature_and_labels_cache = {}
    sampler_options = {'sampler_type': 'batched_balanced_sampler'}

    data_loader_options = {'train_batch_size': train_batch_size, 'test_batch_size': test_batch_size,
                           'train_sampler_opt': sampler_options, 'balance_dataset': True}

    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    w2v_dim = attribute_descriptors.shape[1]

    model_constructor = functools.partial(DazleFakeComposerUnlimited, w2v_dim=w2v_dim, lambda_=(0, 0),
                                          init_w2v_att=attribute_descriptors, num_decoders=(2, 0),
                                          use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                          dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                          vis_drop_rate=vis_drop_rate, backbone=None)

    optimizer_constructor = functools.partial(optim.RMSprop, lr=pre_lr, weight_decay=1e-4, momentum=momentum)
    scheduler_constructor = None

    optimizer_options = {'optimizer_constructor': optimizer_constructor,
                         'scheduler_constructor': scheduler_constructor}

    experiment_desc = f'{dataset_name} b {train_batch_size} hp search'

    save_folder = f'./final_results/check_backbones/'
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']

    # validation -- runs the training on 3 folds
    val_results = validate_backbones(model_constructor, backbones, [1, 2, 3], dataset_name,
                                     data_loader_options, num_epochs=30,
                                     run_tag=experiment_desc,
                                     log_file=f'log_composer_{attribute_type}_',
                                     exp_metrics=metrics_headers, optimizers_options=optimizer_options,
                                     save_folder=save_folder)

    best_hyper, best_H_val, val_res = val_results
    best_bias = np.mean(val_res[best_hyper]['bias'])
    print(f'best bias is: {best_bias}')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    feature_types = ['densenet_201_T1', 'densenet_201_D2', 'densenet_201_T2', 'densenet_201_D3',
                     'densenet_201_T3', 'densenet_201_D4', 'resnet_101_L1', 'resnet_101_L2', 'resnet_101_L3',
                     'resnet_101_L4']

    parser = argparse.ArgumentParser(description='Welcome to the backbone experiment reproduction script.'
                                                 'You can choose the dataset to train on and the backbone to use with '
                                                 'flags -d and -f respectively.'
                                                 'results will be at ./final_results/check_backbones/')

    parser.add_argument('-d', dest='dataset', type=str, choices=['CUB', 'AWA2', 'SUN'], help='Dataset Name to use.')
    parser.add_argument('-f', dest='features', type=str, choices=feature_types,
                        help='backbone to use.')

    parser.add_argument('-btr', dest='train_batch_size', default=32, type=int, help='training batch size')

    parser.add_argument('-g', dest='gpu_idx', default=0, type=int,
                        help='gpu index (if no gpu is available will run on cpu)')

    opts = parser.parse_args()
    check_backbones(opts.dataset, opts.features, train_batch_size=opts.train_batch_size, gpu_idx=opts.gpu_idx)

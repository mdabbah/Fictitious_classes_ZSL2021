import argparse
import functools
import gc
import itertools
import os

import numpy as np
import torch
from torch import optim

from main_code.models import DazleFakeComposerUnlimited
from main_code.train_eval_CC import plot_validation_pickle, plot_validation_per_fold
from misc import data_utils
from misc.custome_dataset import get_base_transforms, get_testing_transform
from misc.data_loader import ZSLDatasetEmbeddings
from misc.data_utils import load_attribute_descriptors, load_pickle, save_pickle, average_over_folds
from misc.log_utils import Logger
from misc.train_eval_utils import eval_expandable_model, train_model, create_data_loader, freeze_model, \
    get_trainable_params, unfreeze_model


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


def do_valid_hp_search(dataset_name, feature_type, balance_dataset=True, gpu_idx=0, train_batch_size=32,
                       test_batch_size=128):
    base_network = feature_type
    attribute_type = 'bert'

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


def validate_deep_dropout_composer(dataset_name, feature_type, best_hp, train_batch_size=32, gpu_idx=0,
                                   test_batch_size=128, num_epochs=30, num_main_epochs=50, pre_lr=1e-4, main_lr=1e-5):
    gc.collect()
    torch.cuda.empty_cache()

    base_network = feature_type
    attribute_type = 'bert'

    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05
    debug = False

    use_random_erase = False
    best_bias = None
    do_validation = True

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size

    # hyper parameters
    vis_drop_rate, num_fake_in_batch = best_hp

    momentum = 0.9 if dataset_name != 'AWA2' else 0.

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

    optimizer_constructor = functools.partial(optim.RMSprop, lr=pre_lr, weight_decay=1e-4, momentum=momentum)
    main_optimizer_constructor = functools.partial(optim.AdamW, lr=main_lr, weight_decay=1e-4)
    scheduler_constructor = None

    optimizer_options = {'warmup_optimizer_constructor': optimizer_constructor,
                         'main_optimizer_constructor': main_optimizer_constructor,
                         'scheduler_constructor': scheduler_constructor}

    # logging

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

    optimizer = optimizer_constructor(params=learnable_params)
    scheduler = None
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
    optimizer = main_optimizer_constructor(params=learnable_params)
    scheduler = scheduler_constructor(optimizer=optimizer) if scheduler_constructor is not None else None
    best_metrics = train_model(model, optimizer=optimizer, scheduler=scheduler, dataset_loaders=data_loaders,
                               loss_fn=loss_fn, metrics_fn=None, exp_logger=exp_logger, best_metrics=best_metrics,
                               model_path=model_path_main_train, device=device, num_epochs=num_main_epochs,
                               custom_eval_epoch=custom_eval)

    best_metrics['base_network'] = base_network
    best_metrics['dataset_name'] = dataset_name
    best_metrics['experiment_desc'] = f'{experiment_desc}_{second_desc}'
    global_log.log(best_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Welcome to the results reproduction script.'
                                                 'You can choose the dataset to train on and the backbone to use with '
                                                 'flags -d and -f respectively.'
                                                 'By default (if -n flag was not passed) the script will run a '
                                                 'hyper-parameter search first to find the best hyper-parameters '
                                                 '(m=best number of fictitious classes, p=the best drop probability). '
                                                 'After the hyper-parameter search is done the script will continue to '
                                                 'train the full model with the found hyper-parameters.'
                                                 'You can find a list of other parameters bellow.')

    parser.add_argument('-d', dest='dataset', type=str, choices=['CUB', 'AWA2', 'SUN'], help='Dataset Name to use.')
    parser.add_argument('-f', dest='features', type=str, choices=['densenet_201_T3', 'resnet_101_L3', 'resnet_101_L4'],
                        help='backbone to use.')

    parser.add_argument('-n', dest='no_fictitious', action='store_true',
                        help='does not do a hyper-parameter search. i.e. trains with (m=0, p=0.).')

    parser.add_argument('-hp', metavar=('m', 'p'), default=None, nargs=2, type=float,
                        help='run the script with the given hyper-parameters.'
                             'm= number of fictitious classes, p= drop probability.'
                             'pass them as follows: -hp m p')

    parser.add_argument('-btr', dest='train_batch_size', default=32, type=int, help='training batch size')
    parser.add_argument('-bte', dest='test_batch_size', default=128, type=int, help='test batch size')

    parser.add_argument('-we', dest='num_warmup_epochs', default=30, type=int,
                        help='the number of epochs to train with the backbone frozen (first training stage).')
    parser.add_argument('-e', dest='num_main_epochs', default=50, type=int,
                        help='the number of epochs to train with the backbone after warm-up (second training stage).')

    parser.add_argument('-wlr', dest='warmup_lr', default=1e-4, type=float,
                        help='learning rate in the warm-up stage (first training stage).')
    parser.add_argument('-lr', dest='main_lr', default=1e-5, type=float,
                        help='learning rate after the warm-up stage (second training stage).')

    parser.add_argument('-g', dest='gpu_idx', default=0, type=int,
                        help='gpu index (if no gpu is available will run on cpu)')

    opts = parser.parse_args()

    dataset = opts.dataset
    features = opts.features
    if opts.no_fictitious:
        validate_deep_dropout_composer(dataset_name=dataset, feature_type=features, best_hp=(0., 0),
                                       train_batch_size=opts.train_batch_size, gpu_idx=opts.gpu_idx,
                                       test_batch_size=opts.test_batch_size, num_epochs=opts.num_warmup_epochs,
                                       num_main_epochs=opts.num_main_epochs, pre_lr=opts.warmup_lr,
                                       main_lr=opts.main_lr)

    else:

        if opts.hp is None:
            pkl_path, val_res, best_hp = do_valid_hp_search(dataset, feature_type=features, balance_dataset=True,
                                                            gpu_idx=opts.gpu_idx,
                                                            test_batch_size=opts.test_batch_size,
                                                            train_batch_size=opts.train_batch_size)

            plot_validation_pickle(pkl_path)
            for fold in range(1, 4):
                plot_validation_per_fold(pkl_path, fold)
        else:
            best_hp = int(opts.hp[0]), float(opts.hp[1])

        validate_deep_dropout_composer(dataset_name=dataset, feature_type=features, best_hp=best_hp,
                                       train_batch_size=opts.train_batch_size, gpu_idx=opts.gpu_idx,
                                       test_batch_size=opts.test_batch_size, num_epochs=opts.num_warmup_epochs,
                                       num_main_epochs=opts.num_main_epochs, pre_lr=opts.warmup_lr,
                                       main_lr=opts.main_lr)

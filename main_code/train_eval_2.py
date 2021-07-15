import copy
import functools
import gc
import itertools
import os
from glob import glob

import numpy as np
import torch
from torch import optim

from misc import data_utils
from misc.custome_dataset import get_base_transforms, get_testing_transform
from misc.data_utils import load_attribute_descriptors
from misc.log_utils import Logger
from misc.train_eval_utils import create_data_loader, train_model, freeze_model, eval_expandable_model, unfreeze_model, \
    load_model, eval_ensemble
from zsl_feature_composition.ablation_study import cub_with_std_err
from zsl_feature_composition.models import DazleFakeComposerUnlimited, DazleFakeComposer
from zsl_feature_composition.train_eval_CC import validate_deep_model

import pandas as pd


def continue_full_train(dataset_name, feature_type, best_hp, train_batch_size=32, gpu_idx=0):
    gc.collect()
    torch.cuda.empty_cache()
    base_network = feature_type
    attribute_type = 'bert'
    test_batch_size = 128
    num_main_epochs = 20
    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05
    debug = False

    use_random_erase = False

    use_unlimited = True

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size

    # put your hyper parameters here
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
    save_folder = f'./final_results/awa/continue_train_densenet/'
    # save_folder = f'./tmp'
    os.makedirs(save_folder, exist_ok=True)
    model_path_pre_train = os.path.join(save_folder,
                                        f'composer_{dataset_name}_{base_network}_{attribute_type}_{experiment_desc}_pre_train.pth')

    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']

    global_log_path = os.path.join(save_folder, f'global_log_exp_fake_deep_comp_end2end.csv')
    global_log = Logger(global_log_path, ['dataset_name', 'base_network', 'epoch', 'experiment_desc', *metrics_headers],
                        False)

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

    pretrain_path = './final_results/awa/continue_train_densenet/composer_AWA2_densenet_201_T3_bert_b 32 m 8 - p 0.5_pre_train.pth'
    model, saved_dict = load_model(model_path=pretrain_path, model=model, print_stats=True)

    best_bias = saved_dict['stats']['bias']
    best_metrics = saved_dict['stats']
    del saved_dict
    custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=True, fixed_bias=best_bias)

    lam = 0.1
    second_desc = f'finetuned with l2 cnn reg real {lam}'
    model_path_main_train = os.path.join(save_folder,
                                         f'composer_{base_network}_{attribute_type}_{experiment_desc}_{second_desc}.pth')

    log_file = os.path.join(save_folder,
                            f'log_composer_{base_network}_{attribute_type}_{experiment_desc}_{second_desc}.csv')
    exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

    unfreeze_model(model.backbone)
    model.save_backbone_chkpnt()
    model.lambda_ = (0, lam)
    learnable_params = [p for p in model.parameters() if p.requires_grad]
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


def run_end2end_hp_search(dataset_name, feature_type, best_hp, train_batch_size=32, gpu_idx=0):
    # dataset_name = 'AWA2'
    gc.collect()
    torch.cuda.empty_cache()
    # feature_type = 'resnet_101_L4'  # 'densenet_201_T3' , 'resnet_101_L4', 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'bert'
    # train_batch_size = 32
    test_batch_size = 128
    num_epochs = 30
    num_main_epochs = 10
    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05
    use_offset_cal = True
    debug = False

    use_random_erase = False
    best_bias = None
    do_validation = False

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
    experiment_desc = f'{dataset_name} b {train_batch_size} hp search'

    save_folder = f'./final_results/awa/exp_unlmtd_fake_deep_comp_validate_end2end_{dataset_name}_{feature_type}_{experiment_desc}/'
    # save_folder = f'./tmp'
    metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']

    # validation
    if do_validation:
        val_results = validate_deep_model(model_constructor, [[4, 8, 15, 25], [0.1, 0.25, 0.5, 0.75, 0.9]], [1, 2, 3],
                                          dataset_name,
                                          data_loader_options, num_warmup_epochs=num_epochs, num_epochs=10,
                                          run_tag=experiment_desc,
                                          log_file=f'log_composer_{base_network}_{attribute_type}_',
                                          exp_metrics=metrics_headers, optimizers_options=optimizer_options,
                                          debug=debug, save_folder=save_folder)

        best_hyper, best_H_val, val_res = val_results
        best_bias = np.mean(val_res[best_hyper]['bias'])
        print(f'best bias is: {best_bias}')
        torch.cuda.empty_cache()

    my_experiments = [(0, 0.)] + list(itertools.product([4, 8, 15, 25], [0.1, 0.25, 0.5, 0.75, 0.9]))
    num_exps = len(my_experiments)
    for i, hp in enumerate(list(my_experiments)):

        num_fake_in_batch, vis_drop_rate = hp
        print(f'exp num {i}/{num_exps}  m={num_fake_in_batch}, p={vis_drop_rate}')

        experiment_desc = f'{dataset_name} b {train_batch_size} m {num_fake_in_batch} - p {vis_drop_rate}'

        print(experiment_desc)
        os.makedirs(save_folder, exist_ok=True)
        model_path_pre_train = os.path.join(save_folder,
                                            f'composer_{dataset_name}_{base_network}_{attribute_type}_{experiment_desc}_pre_train.pth')

        log_file = os.path.join(save_folder, f'log_composer_{base_network}_{attribute_type}_{experiment_desc}.csv')
        exp_logger = Logger(log_file, ['epoch', 'test_loss', *metrics_headers], overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_exp_fake_deep_comp_end2end.csv')
        global_log = Logger(global_log_path, ['dataset_name', 'base_network', 'experiment_desc', *metrics_headers],
                            False)
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

        custom_eval = functools.partial(eval_expandable_model, use_post_bias_correction=use_offset_cal,
                                        fixed_bias=best_bias)

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


def ensemble(dataset_name, feature_type, train_batch_size=32, gpu_idx=0, hp=(0., 0), use_sftmx=False):
    attribute_type = 'bert'
    test_batch_size = 128
    bias = 0
    lam = 0.0
    use_dropout = False
    dropout_rate = 0.05
    use_random_erase = False

    use_unlimited = True

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size
    debug = False
    device = torch.device("cuda:{}".format(gpu_idx) if torch.cuda.is_available() else "cpu")
    # put your hyper parameters here
    vis_drop_rate, num_fake_in_batch = hp

    data_utils.feature_and_labels_cache = {}
    num_workers = 0 if debug else 4
    sampler_options = {'sampler_type': 'batched_balanced_sampler'}
    train_transforms = get_base_transforms(input_img_size, use_random_erase)
    test_transforms = get_testing_transform(input_img_shape)

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
    attribute_descriptors = load_attribute_descriptors(attribute_type, dataset_name)
    w2v_dim = attribute_descriptors.shape[1]

    # building model
    seed = 214  # 215#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    lambda_ = (lam, 0)
    model_constructor = DazleFakeComposer
    if use_unlimited:
        model_constructor = DazleFakeComposerUnlimited
    model_pre = model_constructor(features_dim=img_shape, w2v_dim=w2v_dim, lambda_=lambda_,
                                  init_w2v_att=attribute_descriptors, classes_attributes=class_defs,
                                  seen_classes=seen_classes, unseen_classes=unseen_classes, num_decoders=(2, 0),
                                  use_dropout=use_dropout, device=device, bias=bias, cal_unseen=True,
                                  dropout_rate=dropout_rate, num_fake_in_batch=num_fake_in_batch,
                                  vis_drop_rate=vis_drop_rate, backbone=feature_type)

    model_pre.to(device)

    model_path_template = f'./final_results/{dataset_name.lower()}/exp_unlmtd_fake_deep_comp_validate_end2end_' \
                          f'{dataset_name}_{feature_type}_{dataset_name} b 32 m {num_fake_in_batch} - ' \
                          f'p {vis_drop_rate}/'
    model_paths = glob(model_path_template + '*pth')

    load_model(model_paths[0], model_pre, print_stats=True)
    model_ft = copy.deepcopy(model_pre)
    model, saved_dict = load_model(model_paths[1], model_ft, print_stats=True)
    ft_bias = saved_dict['stats']['bias']
    del saved_dict
    gc.collect()
    torch.cuda.empty_cache()

    pre_bias = get_pre_bias(model_path_template)

    print('fixed bias')
    for alpha in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]:
        res = eval_ensemble([model_pre, model_ft], [pre_bias, ft_bias], [test_seen, test_unseen], alpha,
                            use_sftmx=use_sftmx)
        print('-' * 60)
        print(f'alpha is {alpha}')
        print(res)

    print('-' * 120)
    print('-' * 120)
    print('best bias')
    for alpha in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]:
        res = eval_ensemble([model_pre, model_ft], [pre_bias, ft_bias], [test_seen, test_unseen], alpha,
                            use_post_bias_correction=True, use_sftmx=use_sftmx)
        print('-' * 60)
        print(f'alpha is {alpha}')
        print(res)


def get_pre_bias(exp_folder, pre_max_epoch=30):
    validations_folder = os.path.join(exp_folder, 'validations/')
    validations_csvs = glob(validations_folder + '*fold [123].csv')

    data_frames = [pd.read_csv(csv) for csv in validations_csvs]
    biases = [data_frames[i]['bias'][np.argmax(data_frames[i]['H'][:pre_max_epoch])] for i in range(len(data_frames))]

    return np.mean(biases)


def get_bias_curve(exp_folder):
    validations_folder = os.path.join(exp_folder, 'validations/')
    validations_csvs = glob(validations_folder + '*fold [123].csv')

    data_frames = [pd.read_csv(csv) for csv in validations_csvs]
    biases = np.array([data_frames[i]['bias'] for i in range(len(data_frames))])

    return np.mean(biases, axis=0)


def get_exp_folder(dataset_name, feature_type, num_fake_in_batch, vis_drop_rate):
    return f'./final_results/{dataset_name.lower()}/exp_unlmtd_fake_deep_comp_validate_end2end_' \
           f'{dataset_name}_{feature_type}_{dataset_name} b 32 m {num_fake_in_batch} - ' \
           f'p {vis_drop_rate}/'


class Counter():
    i = 0


def validate_deep_dropout_composer_wbias_curve(dataset_name, feature_type, best_hp, train_batch_size=32, gpu_idx=0):
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
    do_validation = False

    use_unlimited = True

    input_img_size = 324
    input_img_shape = input_img_size, input_img_size

    # put your hyper parameters here
    num_fake_in_batch = 8  # for CUB:= 70, for AWA2:= 8
    vis_drop_rate = 0.5  # for CUB:= 0.5, for AWA2:= 0.5
    vis_drop_rate, num_fake_in_batch = best_hp

    exp_folder = get_exp_folder(dataset_name, feature_type, num_fake_in_batch, vis_drop_rate)
    bias_curve = get_bias_curve(exp_folder)

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

    experiment_desc = f'{dataset_name} b {train_batch_size} m {num_fake_in_batch} - p {vis_drop_rate} -- with bias curve training'
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
    save_folder = f'./final_results/debug_{dataset_name.lower()}/exp_unlmtd_fake_deep_comp_validate_end2end_{dataset_name}_{feature_type}_{experiment_desc}/'
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

    custom_eval_ = functools.partial(eval_expandable_model, use_post_bias_correction=True)

    def custom_eval(m, l, loss_fn=loss_fn, metric2monitor='H', counter={'i': 0}):
        # counter = Counter()
        res = eval_expandable_model(m, l, use_post_bias_correction=True, fixed_bias=bias_curve[counter['i']],
                                    metric2monitor=metric2monitor)
        counter['i'] += 1
        print(f'counter is: {counter["i"]}')
        return res

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


if __name__ == '__main__':
    # for ds, ar in itertools.product(['AWA2', 'SUN'], ['resnet_101_L4', 'resnet_101_L3', 'densenet_201_T3']):
    #     create_CF_graphs(ds, ar)
    cub_with_std_err()
    # for ds in ['AWA2', 'SUN', 'CUB'][2:]:
    #     create_ablation_table(ds)
    # # continue_full_train('AWA2', 'densenet_201_T3', (0.5, 8), gpu_idx=1)
    # # run_end2end_hp_search('AWA2', 'densenet_201_T3', (None, None), train_batch_size=32, gpu_idx=0)
    #
    # dataset_name = 'AWA2'
    # feature_type = 'resnet_101_L3'

    # ensemble(dataset_name, feature_type, train_batch_size=32, gpu_idx=1, hp=(0.5, 15), use_sftmx=False)

    # validate_deep_dropout_composer_wbias_curve(dataset_name, feature_type, (0.5, 15), gpu_idx=0)

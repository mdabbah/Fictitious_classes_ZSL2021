import copy
import functools
import gc
import itertools
import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from plotly import express as px
from torch import optim

from misc import data_utils
from misc.data_loader import ZSLDatasetEmbeddings

from misc.data_utils import load_pickle, load_attribute_descriptors
from misc.log_utils import Logger
from misc.train_eval_utils import create_data_loader, eval_expandable_model, train_model
from zsl_feature_composition.models import DazleFakeComposerUnlimited
from zsl_feature_composition.train_eval_CC import log_val_res_pkl


def get_fold(file_path):
    return int(file_path.split('fold ')[1][:-4])


def is_ours(file_path):
    return 'm 0 - p 0.0' not in file_path


def create_CF_graphs(dataset_name, arch):
    save_folder = f'./catastrophic_forgetting_graphs/{dataset_name}/'
    os.makedirs(save_folder, exist_ok=True)
    fold_files = glob(f'./final_results/{dataset_name.lower()}/*{arch}*/validations/*fold *.csv')

    arch = arch.replace('_', ' ')

    for f in fold_files:
        if 'copy' in f:
            continue
        fold_id = get_fold(f)
        with_fictitious = is_ours(f)

        df = pd.read_csv(f)
        original_headers = ['acc_zs', 'acc_novel', 'acc_seen', 'H', 'bias']
        new_headers = ['T1', 'accuracy unseen', 'accuracy seen', 'H', 'bias']
        rename_dict = {k: v for k, v in zip(original_headers, new_headers)}

        df = df.rename(columns=rename_dict)
        fig = px.line(df, x=range(len(df)), y=new_headers)
        fictitious_str = 'with fictitious classes' if with_fictitious else 'w.o. fictitious classes'
        figure_path = os.path.join(save_folder, f'{arch}  {fictitious_str} validation fold {fold_id}.html')
        fig.update_layout(title=f'{dataset_name} {arch}  {fictitious_str} validation fold {fold_id}',
                          xaxis_title='epoch', yaxis_title='')
        fig.write_html(figure_path)
        print(f'created{figure_path}')


get_hp = lambda s: s.split(' m ')[1].split('/val')[0]
get_arch = lambda s: s.split('composer_')[1].split('_bert')[0]
get_cut_idx = lambda df: np.argmin(df['epoch'][1:]) + 1


def create_results_table(dataset):
    save_path = './final_results/reproduce/results_table.csv'
    files = glob(f'./final_results/reproduce/{dataset.lower()}/*/log*.csv')
    metrics_headers = ['epoch', 'test_loss', 'acc_novel', 'acc_seen', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']
    headers = ['dataset_name', 'backbone', 'with_fictitious', 'finetuned', * metrics_headers]
    save_folder ='./final_results/reproduce'
    os.makedirs(save_folder, exist_ok=True)
    logger = Logger(save_path, headers, False)
    for f in files:
        arch = get_arch(f)
        hp = get_hp(f)
        method = (not '0 - p 0.0' in f)
        print(f)
        df = pd.read_csv(f)
        cut_idx = get_cut_idx(df)
        regular_res = df.iloc[np.argmax(df['H'][:cut_idx])].to_dict()
        ft_res = df.iloc[cut_idx + np.argmax(df['H'][cut_idx:])].to_dict()
        assert len(df) == 80

        # metrics_headers = b.keys().to_list()

        regular_res['dataset_name'] = dataset
        regular_res['backbone'] = arch
        regular_res['with_fictitious'] = method
        regular_res['finetuned'] = False
        logger.log(regular_res)

        ft_res['dataset_name'] = dataset
        ft_res['backbone'] = arch
        ft_res['with_fictitious'] = method
        ft_res['finetuned'] = True
        logger.log(ft_res)



def create_ablation_table(dataset):

    save_folder = './final_results/reproduce/ablation_study_stderr/'
    val_res_pth = f'{save_folder}/val_res_{dataset}.pkl'

    metrics_headers = ['epoch', 'test_loss', 'acc_novel', 'acc_seen', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']
    headers = ['dataset_name', 'backbone', 'with_fictitious', 'finetuned', * metrics_headers]

    os.makedirs(save_folder, exist_ok=True)
    logger = Logger(f'{save_folder}/ablation_study_{dataset}_reproduce.csv', headers, True)

    files = glob(f'./final_results/reproduce/{dataset.lower()}/*/validations/*fold *.csv')
    for f in files:
        arch = get_arch(f)
        hp = get_hp(f)
        method = '0 - p 0.0' != hp

        df = pd.read_csv(f)
        cut_idx = get_cut_idx(df)
        regular_res = df.iloc[np.argmax(df['H'][:cut_idx])].to_dict()
        ft_res = df.iloc[cut_idx + np.argmax(df['H'][cut_idx:])].to_dict()

        assert len(df) == 80

        # metrics_headers = b.keys().to_list()
        exp_key = (arch, method, 'frozen')
        log_val_res_pkl(val_res_pth, exp_key, regular_res, metrics_headers)

        exp_key = (arch, method, 'ft')
        log_val_res_pkl(val_res_pth, exp_key, ft_res, metrics_headers)

        regular_res['dataset_name'] = dataset
        regular_res['backbone'] = arch
        regular_res['with_fictitious'] = method
        regular_res['finetuned'] = False
        logger.log(regular_res)

        ft_res['dataset_name'] = dataset
        ft_res['backbone'] = arch
        ft_res['with_fictitious'] = method
        ft_res['finetuned'] = True
        logger.log(ft_res)

    val_res = load_pickle(val_res_pth)
    logger = Logger(f'{save_folder}/ablation_study_wstderr_reproduce.csv', headers, False)
    for exp, exp_dict in val_res.items():

        stderr_dict = {k:  np.std(np.mean(v) * 100) / np.sqrt(len(v)) for k, v in exp_dict.items()}
        stderr_dict = {k:  np.mean(v) * 100 for k, v in exp_dict.items()}

        exp_dict_avg = {k: f'{np.mean(v) * 100: .2f} \pm {np.std(np.array(v) * 100) / np.sqrt(len(v)):.2f}' for k, v in
                        exp_dict.items()}
        exp_dict_avg['backbone'] = exp[0]
        exp_dict_avg['dataset_name'] = dataset
        exp_dict_avg['with_fictitious'] = exp[1]
        exp_dict_avg['finetuned'] = exp[2]
        logger.log(exp_dict_avg)


def cub_with_std_err():
    save_folder = './final_results/ablation_study_stderr_cub/'
    val_res = load_pickle('./ablation_study_cub/val_res.pkl')
    os.makedirs(save_folder, exist_ok=True)

    metrics_headers = ['epoch', 'test_loss', 'acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'bias', 'auc']
    headers = ['dataset_name', 'backbone', 'with_fictitious', 'finetuned', * metrics_headers]

    logger = Logger(f'{save_folder}/ablation_study_wstderr_2.csv', headers, False)
    val_res_2 = {(k[0], k[1], k[2], False): v for k,v in val_res.items() if k[1] =='bert' and k[2]==2}
    a = copy.deepcopy(val_res_2[('densenet_201_T3', 'bert', 2, False)])
    val_res_2[('densenet_201_T3', 'bert', 2, False)] = {k: v[:3] for k, v in a.items()}
    val_res_2[('densenet_201_T3', 'bert', 2, True)] = {k: v[3:] for k, v in a.items()}
    val_res_2[('resnet_101_L4', 'bert', 2, False)] = {k: v[:3] for k, v in val_res_2[('resnet_101_L4', 'bert', 2, False)].items()}
    for exp, exp_dict in val_res_2.items():

        stderr_dict = {k:  np.std(np.mean(v) * 100) / np.sqrt(len(v)) for k, v in exp_dict.items()}
        stderr_dict = {k:  np.mean(v) * 100 for k, v in exp_dict.items()}

        exp_dict_avg = {k: f'{np.mean(v) * 100: .2f} +- {np.std(np.array(v) * 100) / np.sqrt(len(v)):.2f}' for k, v in
                        exp_dict.items()}
        exp_dict_avg['backbone'] = exp[0]
        exp_dict_avg['dataset_name'] = 'CUB'
        exp_dict_avg['with_fictitious'] = exp[3]
        exp_dict_avg['finetuned'] = False
        logger.log(exp_dict_avg)


def ablation_study():
    dataset_name = 'AWA2'
    feature_type = 'resnet_101_L4'  # 'resnet_101_L4', 'densenet_201_T3' , 'resnet_101_L3'
    base_network = feature_type
    attribute_type = 'w2v'
    train_batch_size = 50
    test_batch_size = 64
    num_epochs = 30
    bias = 1
    lam = 0.1
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
        itertools.product(['densenet_201_T3', 'resnet_101_L3', 'resnet_101_L4'], [False], ['w2v', 'bert'],
                          [0.0], [32], [(0., 0)], [True], [1, 2, 3], [1, 2, 3]))

    # my_last_exp = list(
    #     itertools.product(['densenet_201_T3'], [False], ['bert'],
    #                       [0.9], [32], [(0.5, 70)], [True], [2,], [1, 2, 3]))

    # my_experiments.extend(my_last_exp)
    norm_class_defs, use_log, scale, offset, log_base = True, False, 1, 0, 0

    curr = 'resnet_101_L4'
    x = 0
    for exp_num, exp_params in list(enumerate(my_experiments))[18:]:
        vis_drop_rate, num_fake_in_batch = 0., 0

        feature_type, norm_v, attribute_type, momentum, train_batch_size, hp, balance_dataset, num_decoders, \
        fold_id = exp_params
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

        hp_headers = ['attribute_type', 'keep_rate', 'num_fake', 'balanced', 'num_decoders',
                      'fold_id']

        print(f'exp number: {exp_num}')
        print(experiment_desc)
        save_folder = f'./ablation_study_{dataset_name.lower()}/no_cal_loss/'
        os.makedirs(save_folder, exist_ok=True)
        model_path = os.path.join(save_folder, f'df_{base_network}_{attribute_type}_{experiment_desc}.pth')
        print(model_path)

        log_file = os.path.join(save_folder,
                                f'log_df_{base_network}_{attribute_type}_{experiment_desc}_fold {fold_id}.csv')
        val_res_pkl_path = os.path.join(save_folder, f'val_res.pkl')
        metrics_headers = ['acc_seen', 'acc_novel', 'H', 'acc_zs', 'supervised_acc', 'auc']
        exp_headers = ['epoch', 'test_loss', *metrics_headers]
        exp_logger = Logger(log_file, exp_headers, overwrite=False)

        global_log_path = os.path.join(save_folder, f'global_log_file_deep_comp.csv')
        global_log = Logger(global_log_path,
                            ['dataset_name', 'base_network', 'epoch', *hp_headers, 'experiment_desc', *metrics_headers],
                            False)

        # data loading
        num_workers = 4
        create_data_loader(dataset_name, 'train_gzsl', train_batch_size, num_workers, feature_type=feature_type,
                           fold_id=fold_id, offset=0, )
        train_loader = ZSLDatasetEmbeddings(dataset_name, feature_type, f'train_gzsl', train_batch_size,
                                            conv_features=True, split_id=fold_id,
                                            offset=0, shuffle_batch=True, balance_dataset=balance_dataset).to(device)

        instance_shape = train_loader[0][0].shape
        shape_str = f'[{instance_shape[1]}x{instance_shape[2]}x{instance_shape[3]}]'
        print(shape_str)

        test_seen = ZSLDatasetEmbeddings(dataset_name, feature_type, f'val_seen', test_batch_size,
                                         conv_features=True,
                                         split_id=fold_id,
                                         offset=0).to(device)
        seen_classes = test_seen.classes
        test_unseen = ZSLDatasetEmbeddings(dataset_name, feature_type, f'val_unseen', test_batch_size,
                                           conv_features=True, split_id=fold_id,
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

        best_metrics['attribute_type'] = attribute_type
        best_metrics['keep_rate'] = vis_drop_rate
        best_metrics['num_fake'] = num_fake_in_batch
        best_metrics['balanced'] = balance_dataset

        best_metrics['num_decoders'] = num_decoders
        best_metrics['fold_id'] = fold_id

        exp_key = feature_type, attribute_type, num_decoders, vis_drop_rate, num_fake_in_batch
        log_val_res_pkl(val_res_pkl_path, exp_key, best_metrics, exp_headers)

        global_log.log(best_metrics)

    val_res = load_pickle(val_res_pkl_path)  # type Dict
    ablation_table = Logger(f'{save_folder}/ablation_table.csv', ['base_network', 'attribute_type', 'num_decoders',
                                                                  'drop_rate', 'num_fake',
                                                                  *metrics_headers], overwrite=False)

    for exp, exp_dict in val_res.items():
        exp_dict_avg = {k: np.mean(v) for k, v in exp_dict.items()}
        exp_dict_avg['base_network'] = exp[0]
        exp_dict_avg['attribute_type'] = exp[1]
        exp_dict_avg['num_decoders'] = exp[2]
        exp_dict_avg['drop_rate'] = exp[3]
        exp_dict_avg['num_fake'] = exp[4]
        ablation_table.log(exp_dict_avg)

if __name__ == '__main__':
    for ds in ['CUB', 'AWA2', 'SUN']:
        # create_ablation_table(ds)
        create_results_table(ds)


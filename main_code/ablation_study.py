import os
from glob import glob

import numpy as np
import pandas as pd

from plotly import express as px

from misc.data_utils import load_pickle
from misc.log_utils import Logger

from main_code.reproduce import log_val_res_pkl


def get_fold(file_path):
    return int(file_path.split('fold ')[1][:-4])


def is_ours(file_path):
    return 'm 0 - p 0.0' not in file_path


def create_CF_graphs(dataset_name, arch):
    save_folder = f'./catastrophic_forgetting_graphs/{dataset_name}/'
    os.makedirs(save_folder, exist_ok=True)
    fold_files = glob(f'./final_results/reproduce/{dataset_name.lower()}/*{arch}*/validations/*fold *.csv')

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
    headers = ['dataset_name', 'backbone', 'with_fictitious', 'finetuned', *metrics_headers]
    save_folder = './final_results/reproduce'
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
    headers = ['dataset_name', 'backbone', 'with_fictitious', 'finetuned', *metrics_headers]

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
        stderr_dict = {k: np.std(np.mean(v) * 100) / np.sqrt(len(v)) for k, v in exp_dict.items()}
        stderr_dict = {k: np.mean(v) * 100 for k, v in exp_dict.items()}

        exp_dict_avg = {k: f'{np.mean(v) * 100: .2f} \pm {np.std(np.array(v) * 100) / np.sqrt(len(v)):.2f}' for k, v in
                        exp_dict.items()}
        exp_dict_avg['backbone'] = exp[0]
        exp_dict_avg['dataset_name'] = dataset
        exp_dict_avg['with_fictitious'] = exp[1]
        exp_dict_avg['finetuned'] = exp[2]
        logger.log(exp_dict_avg)


if __name__ == '__main__':
    for ds in ['CUB', 'AWA2', 'SUN']:
        create_ablation_table(ds)
        create_results_table(ds)

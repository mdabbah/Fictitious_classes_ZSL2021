import os
from glob import glob
from typing import Dict
import numpy as np
import plotly.graph_objects as go
import plotly.offline.offline as py
import pickle

from plotly.subplots import make_subplots

from misc.data_utils import load_pickle
import pandas as pd


def plot_metric_map(data_path, metric, axis_names=('lambda1', 'lambda2')):
    """
    supports only 2 hp maps, plts a heat map from the hp search
    :param data_path: where is the pkl
    :param metric: the key under the results are
    :param axis_names: what axis names to display (dflt:  'lambda1', 'lambda2')
    :return: x_values, y_values, z, data
    """
    data = load_pickle(data_path)
    fig_path = data_path[:-4] + f' {metric}.html'

    x_values = []
    y_values = []
    for (_lambda1, _lambda2) in data.keys():
        if _lambda1 not in x_values:
            x_values.append(_lambda1)

        if _lambda2 not in y_values:
            y_values.append(_lambda2)

    x_values = np.sort(x_values)
    y_values = np.sort(y_values)

    z = np.zeros([len(x_values), len(y_values)])
    for i, _lambda1 in enumerate(x_values):
        for j, _lambda2 in enumerate(y_values):
            if isinstance(data[_lambda1, _lambda2], dict):
                z[i, j] = np.mean(data[_lambda1, _lambda2][metric])
            else:
                z[i, j] = np.mean(data[_lambda1, _lambda2])

    x_values = [str(x) + '_' for x in x_values]
    y_values = [str(y) + '_' for y in y_values]
    fig = go.Figure(data=go.Heatmap(
        z=z.T, x=x_values, y=y_values))

    fig.update_layout(
        title=metric,
        xaxis_title=axis_names[0],
        yaxis_title=axis_names[1])

    py.plot(fig, filename=fig_path, auto_open=True)
    return x_values, y_values, z, data


def plot_metric_map_from_data(data, title, save_folder='./val_results_heatmaps/', metric='H',
                              axis_names=('lambda1', 'lambda2'), transpose=False):
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, title)
    save_path = save_path.split('-')[0] + '.html'

    x_values = []
    y_values = []
    for (_lambda1, _lambda2) in data.keys():
        if _lambda1 not in x_values:
            x_values.append(_lambda1)

        if _lambda2 not in y_values:
            y_values.append(_lambda2)

    x_values = np.sort(x_values)
    y_values = np.sort(y_values)

    z = np.zeros([len(x_values), len(y_values)])
    for i, _lambda1 in enumerate(x_values):
        for j, _lambda2 in enumerate(y_values):
            if (_lambda1, _lambda2) not in data:
                z[i, j] = np.nan
                continue

            if isinstance(data[_lambda1, _lambda2], dict):
                z[i, j] = np.mean(data[_lambda1, _lambda2][metric])
            else:
                z[i, j] = np.mean(data[_lambda1, _lambda2])

    x_values = [str(x) + '_' for x in x_values]
    y_values = [str(y) + '_' for y in y_values]

    axis_values = [x_values, y_values]
    x_axis_idx = 0
    if transpose:
        x_axis_idx = 1
        z = z.T

    fig = go.Figure(data=go.Heatmap(
        z=z.T, x=axis_values[x_axis_idx], y=axis_values[1 - x_axis_idx]))

    fig.update_layout(
        title=title,
        xaxis_title=axis_names[x_axis_idx],
        yaxis_title=axis_names[1 - x_axis_idx])

    py.plot(fig, filename=save_path, auto_open=True)
    return x_values, y_values, z, data


def plot_trace(y, x=None, axis_labels=('x', 'y'), title=None, save_folder='./'):
    if x is None:
        x = np.arange(len(y))

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, title + '.html')

    fig = go.Figure(data=go.Scatter(x=x, y=y))
    fig.update_layout(
        title=title,
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1])

    py.plot(fig, filename=save_path, auto_open=False)


def plot_bars(y, x=None, axis_labels=('x', 'y'), title=None, save_folder='./'):
    if x is None:
        x = np.arange(len(y))

    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, title + '.html')

    if isinstance(y, list):
        data = [go.Bar(x=x, y=y_i) for y_i in y]
        fig = go.Figure(data=data)
        fig.update_layout(barmode='group')
    else:
        fig = go.Figure(data=go.Bar(x=x, y=y))

    fig.update_layout(
        title=title,
        xaxis_title=axis_labels[0],
        yaxis_title=axis_labels[1])

    py.plot(fig, filename=save_path, auto_open=False)


def plot_subplots(files, column, rows, cols, save_dir, global_title):
    traces = []
    read_files = []
    titles = []
    for file in files:
        if file in read_files:
            continue

        title = os.path.basename(file.split('novo16')[1].split('fold')[0])
        # base_dir = os.path.dirname(file)
        fold_files = [f for f in files if f.find(title) >= 0]
        read_files.extend(fold_files)
        folds_traces = []
        for ffile in fold_files:
            df = pd.read_csv(ffile)
            trace = np.array(df[column])
            folds_traces.append(trace[np.newaxis, :])

        min_len = np.min([t.shape[1] for t in folds_traces])
        folds_traces = [ft[:, :min_len] for ft in folds_traces]
        trace = np.mean(np.concatenate(folds_traces, axis=0), axis=0)
        traces.append(trace)
        titles.append(title)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)
    for trace_id, trace in enumerate(traces):
        pos = trace_id // cols, trace_id % cols
        fig.add_trace(
            go.Scatter(x=np.arange(len(trace)), y=trace),
            row=pos[0] + 1, col=pos[1] + 1
        )

    save_path = os.path.join(save_dir, global_title) + '.html'
    fig.update_layout(title_text=global_title)
    py.plot(fig, filename=save_path, auto_open=True)
    fig.show()


if __name__ == '__main__':
    title = 'val_res v1 with unseen knowledge + gmm means as attributes + article generation from expert definitions.pkl'
    save_dir = '../DAZLE/visualize_results'
    os.makedirs(save_dir, exist_ok=True)

    files = glob('./models_v5/valid_/*gmm*fold *.csv')
    # plot_subplots(files, 'H', 5, 5, save_dir, 'val_res v2.5 w sim loss4 cosine novo16 H over folds')

    x_values, y_values, z, data = plot_metric_map('./models_v5/valid_fr/' + title, title, save_dir)

    # files = glob('./models_v3/valid/bak/*v3 zero sim*fold*.csv')
    # plot_subplots(files, 'H', 1, 1, save_dir, 'v3 zero simclr s 0.25 mean H over folds')

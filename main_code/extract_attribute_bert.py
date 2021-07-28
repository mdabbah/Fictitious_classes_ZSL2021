# -*- coding: utf-8 -*-

import os
import sys

import torch

from misc.project_paths import DATA_BASE_DIR

pwd = os.getcwd()
sys.path.insert(0, pwd)
# %%
print('-' * 30)
print(os.getcwd())
print('-' * 30)
# %%
import pdb
import pandas as pd
import numpy as np
import pickle
from transformers import BertModel, BertTokenizer
import scipy.io as sio

# %%
dataset_name = 'SUN'
print('Loading pretrain bert model')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,  # Whether the model returns all hidden-states.
                                  )
# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()
dim_w2v = 768
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
print('Done loading model')
# %%
replace_word = []
if dataset_name == 'CUB':
    replace_word = [('spatulate', 'broad'), ('upperparts', 'upper parts'), ('grey', 'gray')]
elif dataset_name == 'AWA2':
    replace_word = [('newworld', 'new world'), ('oldworld', 'old world'), ('nestspot', 'nest spot'),
                    ('toughskin', 'tough skin'),
                    ('longleg', 'long leg'), ('chewteeth', 'chew teeth'), ('meatteeth', 'meat teeth'),
                    ('strainteeth', 'strain teeth'),
                    ('quadrapedal', 'quadrupedal')]
elif dataset_name == 'SUN':
    replace_word = [('rockstone', 'rock stone'), ('dirtsoil', 'dirt soil'), ('man-made', 'man-made'),
                    ('sunsunny', 'sun sunny'),
                    ('electricindoor', 'electric indoor'), ('semi-enclosed', 'semi enclosed'), ('far-away', 'faraway')]

# %%
path = {'CUB': f'{DATA_BASE_DIR}/attribute/CUB/attributes.txt',
        'AWA2': f'{DATA_BASE_DIR}/attribute/AWA2/predicates.txt',
        'SUN': f'{DATA_BASE_DIR}/attribute/SUN/attributes.mat'}[dataset_name]

if dataset_name != 'SUN':
    sep = {'CUB': ' ', 'AWA2': '\t'}['AWA2']
    df = pd.read_csv(path, sep=sep, header=None, names=['idx', 'des'])
    new_des = des = df['des'].values
else:
    matcontent = sio.loadmat(path)
    des = matcontent['attributes'].flatten()
    new_des = [''.join(i.item().split('/')) for i in des]
    df = pd.DataFrame()
# %% filter
if dataset_name == 'CUB':
    new_des = [' '.join(i.split('_')) for i in des]
    new_des = [' '.join(i.split('-')) for i in new_des]
    new_des = [' '.join(i.split('::')) for i in new_des]
    new_des = [i.split('(')[0] for i in new_des]

# %% replace out of dictionary words
for pair in replace_word:
    for idx, sent in enumerate(new_des):
        new_des[idx] = sent.replace(pair[0], pair[1])
print('Done replace OOD words')
# %%
df['new_des'] = new_des
df.to_csv(f'{DATA_BASE_DIR}/attribute/{dataset_name}/new_des.csv')
print('Done preprocessing attribute des')
# %%
all_w2v = []
for sent in new_des:
    print(sent)

    sent = sent.strip()

    input_ids = tokenizer.encode(sent, add_special_tokens=True)
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        hidden_states = model(input_ids)[2]
        token_vecs = torch.stack(hidden_states, dim=0).squeeze()[-2]
        assert token_vecs.size(1) == dim_w2v
        sentence_embedding = torch.mean(token_vecs, dim=0)

        w2v = sentence_embedding.numpy()

    all_w2v.append(w2v[np.newaxis, :])
# %%
all_w2v = np.concatenate(all_w2v, axis=0)
# pdb.set_trace()
# %%
save_path = f'{DATA_BASE_DIR}/w2v/{dataset_name}_attribute_bert.pkl'
with open(f'{DATA_BASE_DIR}/w2v/{dataset_name}_attribute_bert.pkl', 'wb') as f:
    pickle.dump(all_w2v, f)
print(f'saved {save_path}')

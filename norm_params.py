import numpy as np
import pandas as pd
from CMAPSSDataset import CMAPSSDataset

train_df = pd.read_csv('CMAPSSData/train_FD001.txt', sep=' ', header=None)
train_df = train_df.iloc[:, :26]
train_df.columns = CMAPSSDataset.DATASET_COLS
train_dataset = CMAPSSDataset(train_df, sequence_len=30, norm_type='z-score', init=False)
norm_params = CMAPSSDataset.gen_norm_params([train_dataset], 'z-score', False)
np.save('FD001_params.npy', norm_params)

import numpy as np
import random
import os 
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp import autocast

from torchsummary import summary
from sklearn.metrics import r2_score

from ray import tune

import json
import itertools
from itertools import groupby
import gzip 
from io import BytesIO
from time import time 

import matplotlib.pyplot as plt

# import pyBigWig
from scipy.sparse import csc_matrix
import math 

import torch.cuda.amp as amp

import functools

import captum 
from captum.attr import DeepLift, DeepLiftShap
# import seaborn as sns


class DNA_Iter(Dataset):
    def __init__(self, input_name, switch=False, target_window = 128):
        self.target_window = target_window
        self.seq = self.read_memmap_input(input_name)

        self.target_window = target_window
        self.nucs = np.arange(6.)
        self.len = (int(self.seq.shape[0] / (self.target_window)))
        self.switch = switch 
        self.switch_func = np.vectorize(lambda x: x + 1 if (x % 2 == 0) else x - 1)
        self.num_targets = 1

    def __len__(self):
        return self.len 

    def __getitem__(self, idx): 
        seq_subset = self.seq[idx*self.target_window:(idx+1)*self.target_window]
        if self.switch: 
            seq_subset = self.switch_func(list(reversed(seq_subset)))
        dta = self.get_csc_matrix(seq_subset)
        tgt = np.zeros(seq_subset.shape)
        tgt[np.where(seq_subset == 3.)] = 1.
        tgt_window = np.mean(tgt)
        return torch.tensor(dta), torch.tensor(tgt_window)
    
    def read_numpy_input(self, np_gq_name):
        seq = np.load(np_gq_name)
        return seq

    def read_memmap_input(self, mmap_name):
        seq = np.memmap(mmap_name, dtype='float32',  mode = 'r+') #, shape=(2, self.chrom_seq[self.chrom]))
        return seq


    def dna_1hot(self, seq):
        adtensor = np.zeros((4, self.target_window), dtype=float)
        for nucidx in range(len(self.nucs)):
            nuc = self.nucs[nucidx]#.encode()
            j = np.where(seq[0:len(seq)] == nuc)[0]
            adtensor[nucidx, j] = 1
        return adtensor

    def get_csc_matrix(self, seq_subset):
        N, M = len(seq_subset), len(self.nucs)
        dtype = np.uint8
        rows = np.arange(N)
        cols = seq_subset
        data = np.ones(N, dtype=dtype)
        ynew = csc_matrix((data, (rows, cols)), shape=(N, M), dtype=dtype)
        return ynew.toarray()[:, :4]

    def calc_mean_lst(self, lst, n):
        return np.array([np.mean(lst[i:i + n]) for i in range(int(len(lst)/n))])

    
    def slice_arr(self, idx, tgt_mmap, num_targets):
        return torch.tensor(np.nan_to_num(tgt_mmap[idx::int(tgt_mmap.shape[0] / num_targets)].reshape(num_targets, 1)))

    def get_stacked_means(self, idx, tgt_mmap, num_targets):
        vals = map(functools.partial(self.slice_arr, tgt_mmap=tgt_mmap, num_targets=num_targets), np.arange(idx, idx+128))
        stacked_means = torch.stack(list(map(sum, zip(*vals)))) / num_targets
        return stacked_means

    def get_targets(self, idx, tgt_mmap_cl, tgt_mmap_pdx, num_targets_cl, num_targets_pdx):
        stacked_means_cl = self.get_stacked_means(idx, tgt_mmap_cl, num_targets_cl)
        stacked_means_pdx = self.get_stacked_means(idx, tgt_mmap_pdx, num_targets_pdx)
        stacked_full = torch.cat((stacked_means_cl, stacked_means_pdx)).view(stacked_means_cl.shape[0] + stacked_means_pdx.shape[0])
        return stacked_full
    
def make_dsets(input_files_dir):
    cut = .8
    np.random.seed(42)
    chroms_list = [file.split('_')[0] for file in os.listdir(input_files_dir) if file.split('.')[-1] == 'dta']
    np.random.shuffle(chroms_list)
    input_list = np.hstack([[file for file in os.listdir(input_files_dir) if file.split('_')[0] == chrom] for chrom in chroms_list])

    val_input_files = input_list[int(len(input_list)*cut):]

    train_input_files = input_list[:int(len(input_list)*cut)]

    valid_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, val_input_files[i])) for i in range(len(val_input_files))])
    training_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, train_input_files[i]), switch=False) for i in range(len(train_input_files))])
    training_dset_augm = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, train_input_files[i]), switch=True) for i in range(len(train_input_files))])
    return training_dset, valid_dset

def make_loaders(batch_size, training_dset, valid_dset):
    train_loader = DataLoader(dataset=training_dset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(dataset=valid_dset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    return train_loader, val_loader


# Prepare the input 

memmap_data_contigs_dir = os.path.join(os.getcwd(), 'hg38_memmaps')
training_dset, valid_dset = make_dsets(memmap_data_contigs_dir)
train_loader, val_loader = make_loaders(128*8*8, training_dset, valid_dset)


X_tr, y_tr =  next(iter(train_loader))
X_val, y_val =  next(iter(val_loader))

def reshape_input(inp, batch_size):
    return torch.stack(torch.chunk(torch.transpose(inp.reshape(inp.shape[0]*inp.shape[1], 4), 1, 0), batch_size, dim=1)).type(torch.FloatTensor)#.cuda()

X_tr_reshape = reshape_input(X_tr, batch_size=8)
X_val_reshape = reshape_input(X_val, batch_size=8)

# Prepare and load the model 

def load_model(model, filename):
    model.load_state_dict(torch.load(filename))    


model = BasenjiModel(num_targets=1)
load_model(model, "basenji_toy_6char_3epochs.pt")

# Run the DeepLift Algorithm 
dl = DeepLiftShap(model)

attribution = dl.attribute(X_val_reshape, baselines=X_tr_reshape, target=1)
inp_vals = X_val_reshape.detach().numpy()
attr = attribution.detach().numpy()


# (Not an optimal solution) Correspond the DeepLift values to the one-hot encoded nucleotide vector 
nucs_vals = np.zeros((inp_vals.shape[0], inp_vals.shape[-1]))
attr_vals = np.zeros((inp_vals.shape[0], inp_vals.shape[-1]))
for j in range(inp_vals.shape[0]):
    for i in range(len(nucs)):
        ids = np.where(inp_vals[j][i] == 1.)
        nucs_vals[j][ids] = nucs[i]
        for idx in ids: 
            attr_vals[j][idx] = attr[j][i][idx]

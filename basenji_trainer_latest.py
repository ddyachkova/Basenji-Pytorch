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

import pyBigWig
from scipy.sparse import csc_matrix
import math 

import torch.cuda.amp as amp
from basenji_trainer_classes import * 

def main():
    memmap_data_contigs_dir = os.path.join(os.getcwd(), 'hg38_memmaps')
    memmap_data_targets_dir = os.path.join(os.getcwd(), 'hg38_targets_memmaps')
    param_vals = { 
"optimizer" : "Adam", 
"init_lr": 0.001, 
"optimizer_momentum": 0.9, 
"weight_decay": 1e-3, 
"loss": "poisson", 
"num_targets": 3,
# "lambda_param": 0.001, 
# "ltype":1,
# "clip": 2.,
"seq_len": 128*128*8,
"batch_size": 8*2,
"cut": 0.8,
"num_workers": 8,
"num_epochs": 1
}
    model = BasenjiModel(num_targets=3) #debug=False, loss='poisson', num_targets=1, lr=0.01, opt='SGD', momentum=0.99)
    model.compile(device='cuda')
    trainer = Trainer(param_vals, model, memmap_data_contigs_dir, memmap_data_targets_dir)
    trainer.train(debug=False)
    
if __name__ == '__main__':
     main()
    
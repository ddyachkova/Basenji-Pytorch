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


class DNA_Iter(Dataset):
    def __init__(self, input_name, targets_name, num_targets, switch=False, target_window = 128):
        self.target_window = target_window
        self.seq = self.read_memmap_input(input_name)
        self.tgt_mmap = self.read_memmap_input(targets_name)
        self.target_window = target_window
        self.nucs = np.arange(6.)
        self.len = (int(self.seq.shape[0] / (self.target_window)))
        self.switch = switch 
        self.switch_func = np.vectorize(lambda x: x + 1 if (x % 2 == 0) else x - 1)
        self.num_targets = num_targets
    
    def __len__(self):
        return self.len 

    def __getitem__(self, idx): 
        seq_subset = self.seq[idx*self.target_window:(idx+1)*self.target_window]
        if self.switch: 
            seq_subset = self.switch_func(list(reversed(seq_subset)))
        dta = self.get_csc_matrix(seq_subset)
        tgt_window_3 = torch.stack([torch.mean(torch.tensor(np.nan_to_num(self.tgt_mmap[(i*self.seq.shape[0])+(self.target_window*idx): (i*self.seq.shape[0]) +(idx+1)*self.target_window]))) for i in range(self.num_targets)])
        return torch.tensor(dta), tgt_window_3 
    
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

    
class upd_GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(torch.Tensor([1.702]).cuda() * input) * input

def ones_(tensor: Tensor) -> Tensor:
    return torch.ones_like(tensor)

def zeros_(tensor: Tensor) -> Tensor:
    return torch.zeros_like(tensor)

class BasenjiModel(nn.Module):
    def __init__(self, num_targets, n_channel=4, max_len=128, 
                 conv1kc=64, conv1ks=15, conv1st=1, conv1pd=7, pool1ks=8, pool1st=1 , pdrop1=0.4, #conv_block_1 parameters
                 conv2kc=64, conv2ks=5, conv2st=1, conv2pd=3, pool2ks=4 , pool2st=1, pdrop2=0.4, #conv_block_2 parameters
                 conv3kc=round(64*1.125), conv3ks=5, conv3st=1, conv3pd=3, pool3ks=4 , pool3st=1, pdrop3=0.4, #conv_block_2 parameters
                 convdc = 6, convdkc=32 , convdks=3, debug=False):                 
        super(BasenjiModel, self).__init__()
        
        self.convdc = convdc
        self.debug = debug
        self.num_targets =  num_targets
        ## CNN + dilated CNN
        
        self.conv_block_1 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(n_channel, conv1kc, kernel_size=conv1ks, stride=conv1st, padding=conv1pd, bias=False),
            nn.BatchNorm1d(conv1kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool1ks),
            nn.Dropout(p=0.2))
                
        self.conv_block_2 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(conv1kc, conv2kc, kernel_size=conv2ks, stride=conv2st, padding=conv2pd, bias=False),
            nn.BatchNorm1d(conv2kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool2ks),
            nn.Dropout(p=0.2))
        
        self.conv_block_3 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(conv2kc, round(conv2kc*1.125), kernel_size=conv3ks, stride=conv3st, padding=conv3pd, bias=False),
            nn.BatchNorm1d(conv3kc, momentum=0.9, affine=True),
            nn.MaxPool1d(kernel_size=pool3ks),
            nn.Dropout(p=0.2))
        

        self.dilations = nn.ModuleList()
        for i in range(convdc):
            padding = 2**(i)
            self.dilations.append(nn.Sequential(
                upd_GELU(),
                nn.Conv1d(conv3kc, 32, kernel_size=3, padding=padding, dilation=2**i, bias=False),
                nn.BatchNorm1d(32, momentum=0.9, affine=True), 
                upd_GELU(),
                nn.Conv1d(32, 72, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(72, momentum=0.9, affine=True), 
                nn.Dropout(p=0.25)))
            
        self.conv_block_4 = nn.Sequential(
            upd_GELU(),
            nn.Conv1d(72, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(64, momentum=0.9, affine=True), 
            nn.Dropout(p=0.1)) 
            
        self.conv_block_5 = nn.Sequential(
            upd_GELU(),
            nn.Linear(64, self.num_targets, bias=True),
            nn.Softplus(beta=1, threshold=1000)) 

    
        self.conv_block_1[1].weight.data = self.truncated_normal(self.conv_block_1[1].weight, 0.0, np.sqrt(2/60)) #4
        self.conv_block_2[1].weight.data = self.truncated_normal(self.conv_block_2[1].weight, 0.0, np.sqrt(2/322)) # conv1kc
        self.conv_block_3[1].weight.data = self.truncated_normal(self.conv_block_3[1].weight, 0.0, np.sqrt(2/322)) # conv1kc
        self.conv_block_4[1].weight.data = self.truncated_normal(self.conv_block_4[1].weight, 0.0, np.sqrt(2/72)) # 72
        self.conv_block_5[1].weight.data = self.truncated_normal(self.conv_block_5[1].weight, 0.0, np.sqrt(2/64)) # 64        
        self.conv_block_1[2].weight.data = ones_(self.conv_block_1[2].weight)
        self.conv_block_2[2].weight.data = ones_(self.conv_block_2[2].weight)
        self.conv_block_3[2].weight.data = ones_(self.conv_block_3[2].weight)
        self.conv_block_4[2].weight.data = ones_(self.conv_block_4[2].weight)

        
        for i in range(convdc):
            self.dilations[i][1].weight.data = self.truncated_normal(self.dilations[i][1].weight, 0.0, np.sqrt(2/218)) # 72
            self.dilations[i][-2].weight.data = self.truncated_normal(self.dilations[i][-2].weight, 0.0, np.sqrt(2/32)) # 32
            self.dilations[i][2].weight.data = zeros_(self.dilations[i][2].weight)
            self.dilations[i][-2].weight.data = ones_(self.dilations[i][-2].weight)

    
    def truncated_normal(self, t, mean, std):
        torch.nn.init.normal_(t, mean, std)
        while True:
            cond = torch.logical_or(t < (mean - 2.28*std), t > (mean + 2.28*std))
            if not torch.sum(cond):
                break
            t = torch.where(cond, torch.nn.init.normal_(torch.ones(t.shape), mean=mean, std=std), t)
        return t


    def forward(self, seq):
        if self.debug: 
            print (seq.shape)
        seq = self.conv_block_1(seq)
        if self.debug: 
            print ('conv1', seq.shape)
        seq = self.conv_block_2(seq)
        if self.debug: 
            print ('conv2', seq.shape)
        seq = self.conv_block_3(seq)
        if self.debug: 
            print ('conv3', seq.shape)
        for i in range(self.convdc):
            if i == 0:
                y = self.dilations[i](seq)
            if i >= 1:
                y = y.add(self.dilations[i](seq))
            if self.debug: 
                print ('dil', i, self.dilations[i](seq).shape)
        if self.debug:
            print ('y', y.shape)
        res = self.conv_block_4(y)
        if self.debug: 
            print ('res', res.shape)
        res_lin = res.transpose(1, 2)
        if self.debug: 
            print ('res_lin', res_lin.shape)
        res = self.conv_block_5(res_lin)
        if self.debug: 
            print ('res', res.shape)
        return res
        
    def compile(self, device='cpu'):
        self.to(device)
        
        
class Trainer(nn.Module):
    def __init__(self, param_vals, model, memmap_data_contigs_dir, memmap_data_targets_dir):
        super(Trainer, self).__init__()
    
        self.param_vals = param_vals
        self.model = model 
        
        self.train_losses, self.valid_losses, self.train_Rs, self.valid_Rs, self.train_R2, self.valid_R2 = [], [], [], [], [], []
        self.optim_step = 0
        
        
        self.batch_size = self.param_vals.get('batch_size', 8)
        self.num_targets = self.param_vals.get('num_targets', 1)
        self.make_optimizer()
        self.init_loss()
        self.make_dsets(memmap_data_contigs_dir, memmap_data_targets_dir)

#         self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

    def make_optimizer(self): 
        if self.param_vals["optimizer"]=="Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param_vals["init_lr"])
        if self.param_vals["optimizer"]=="AdamW":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.param_vals["init_lr"])
        if self.param_vals["optimizer"]=="SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.param_vals["init_lr"], momentum = self.param_vals["optimizer_momentum"])
        if self.param_vals["optimizer"]=="Adagrad":
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=self.param_vals["init_lr"], weight_decay = self.param_vals["weight_decay"])
    
    def make_dsets(self, input_files_dir, target_files_dir):
        cut = self.param_vals.get('cut', .8)
        np.random.seed(42)
        chroms_list = [file.split('_')[0] for file in os.listdir(target_files_dir)]
        np.random.shuffle(chroms_list)
        input_list = np.hstack([[file for file in os.listdir(input_files_dir) if file.split('_')[0] == chrom] for chrom in chroms_list])
        targets_list = np.hstack([[file for file in os.listdir(target_files_dir) if file.split('_')[0] == chrom] for chrom in chroms_list])

        val_input_files = input_list[int(len(input_list)*cut):]
        val_target_files = targets_list[int(len(targets_list)*cut):]

        train_input_files = input_list[:int(len(input_list)*cut)]
        train_target_files = targets_list[:int(len(targets_list)*cut)]

        self.valid_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, val_input_files[i]), os.path.join(target_files_dir, val_target_files[i]), self.num_targets) for i in range(len(val_input_files))])
        self.training_dset = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, train_input_files[i]), os.path.join(target_files_dir, train_target_files[i]), self.num_targets, switch=False) for i in range(len(train_input_files))])
        self.training_dset_augm = ConcatDataset([DNA_Iter(os.path.join(input_files_dir, train_input_files[i]), os.path.join(target_files_dir, train_target_files[i]),  self.num_targets, switch=True) for i in range(len(train_input_files))])

        
    def make_loaders(self, augm):
        batch_size = int(self.param_vals.get('seq_len', 128*128*8)*self.param_vals.get('batch_size', 8)/128)
        num_workers = self.param_vals.get('num_workers', 8)
        if augm: 
            train_loader = DataLoader(dataset=self.training_dset_augm, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        else: 
            train_loader = DataLoader(dataset=self.training_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        val_loader = DataLoader(dataset=self.valid_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
        return train_loader, val_loader
    
    
    def decayed_learning_rate(self, step, initial_learning_rate, decay_rate=0.96, decay_steps=100000):
        return initial_learning_rate * math.pow(decay_rate, (step / decay_steps))
    
    def upd_optimizer(self, optim_step):
        decayed_lr = self.decayed_learning_rate(optim_step, initial_learning_rate=self.param_vals["init_lr"])
        for g in self.optimizer.param_groups:
            g['lr'] = decayed_lr 

        
    def init_loss(self, reduction="mean"):
        if self.param_vals["loss"]=="mse":
            self.loss_fn = F.mse_loss
        if self.param_vals["loss"]=="poisson":
            self.loss_fn = torch.nn.PoissonNLLLoss(log_input=False, reduction=reduction)
    
    def get_input(self, batch):
        batch_size = self.param_vals.get('batch_size', 8)
        num_targets = self.param_vals.get('num_targets', 1)
        seq_X,y = batch
        X_reshape = torch.stack(torch.chunk(torch.transpose(seq_X.reshape(seq_X.shape[0]*seq_X.shape[1], 4), 1, 0), batch_size, dim=1)).type(torch.FloatTensor).cuda()
#         print (X_reshape.shape, y.shape)
        if X_reshape.shape[-1] == self.param_vals.get('seq_len', 128*128*8):
            y =  torch.stack(torch.chunk(y, batch_size, dim=0)).view(batch_size, 1024, num_targets).type(torch.FloatTensor).cuda()
            y = F.normalize(y, dim=1)            
            return X_reshape, y
        else:
            return np.array([0]), np.array([0])

    def plot_results(self, y, out, num_targets):
        for i in range(num_targets):
            ys = y[:, :, i].flatten().cpu().numpy()
            preds = out[:, :, i].flatten().detach().cpu().numpy()
            plt.plot(np.arange(len(ys.flatten())), ys.flatten(), label='True')
            plt.plot(np.arange(len(preds.flatten())), preds.flatten(), label='Predicted', alpha=0.5)
            plt.legend()
            plt.show()        
    
    def train(self, debug):
#         scaler = torch.cuda.amp.GradScaler(enabled=True)
        for epoch in range(self.param_vals.get('num_epochs', 10)):
            if epoch % 2 == 0: 
                augm = False
            else: 
                augm = True
            train_loader, val_loader = self.make_loaders(augm)
            for batch_idx, batch in enumerate(train_loader):
                print_res, plot_res = False, False
                self.model.train()
                x, y = self.get_input(batch)
                if (debug): 
                    print (x.shape, y.shape)
                if x.shape[0] != 1: 
                    self.optimizer.zero_grad()
                    if batch_idx%10==0:
                        print_res = True
                        if batch_idx%100==0:
                            plot_res = True
                    self.train_step(x, y, print_res, plot_res, epoch, batch_idx, train_loader)
                    print_res, plot_res = False, False
            print(self.train_R2)
                             
            if val_loader:
                print_res, plot_res = False, False
                self.model.eval()
                for batch_idx, batch in enumerate(val_loader):
                    print_res, plot_res = False, False 
                    x, y = self.get_input(batch)
                    if x.shape[0] != 1: 
                        if batch_idx%10==0:
                            print_res = True
                            if batch_idx%100==0:
                                plot_res = True
                        self.eval_step(x, y, print_res, plot_res, epoch, batch_idx, val_loader) 
                        print_res, plot_res = False, False 

            train_arrs = np.array([self.train_losses, self.train_Rs, self.train_R2])
            val_arrs = np.array([self.valid_losses, self.valid_Rs, self.valid_R2])
            self.plot_metrics(epoch+1, train_arrs, val_arrs)



    def train_step(self, x, y, print_res, plot_res, epoch, batch_idx, train_loader):
        with torch.cuda.amp.autocast():
            out = self.model(x).view(y.shape)
            loss = self.loss_fn(out,y)
            if self.param_vals.get('lambda_param', None): 
                loss = self.regularize_loss(self.param_vals["lambda_param"], self.param_vals["ltype"], self.model, loss)
        R, r2 = self.calc_R_R2(y, out, self.num_targets)
        

        loss.backward()
        if self.param_vals.get('clip', None): 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.param_vals["clip"])
        
        self.optimizer.step()
        self.optim_step += 1
        self.upd_optimizer(self.optim_step)
    
        self.train_losses.append(loss.data.item())
        self.train_Rs.append(R.item())
        self.train_R2.append(r2.item())
        if print_res: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}\tR2: {:.6f}'.format(
                          epoch, batch_idx, len(train_loader), int(100. * batch_idx / len(train_loader)),
                          loss.item(), R.item(), r2.item()))
        if plot_res: 
            self.plot_results(y, out, self.num_targets)

    def eval_step(self, x, y, print_res, plot_res, epoch, batch_idx, val_loader):
        out = self.model(x).view(y.shape)
        loss = self.loss_fn(out,y)
        R, r2 = self.calc_R_R2(y, out, self.num_targets)
        self.valid_losses.append(loss.data.item())
        self.valid_Rs.append(R.item())
        self.valid_R2.append(r2.item())                
        
        if print_res: 
            print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tR: {:.6f}\tR2: {:.6f}'.format(
                          epoch, batch_idx, len(val_loader), int(100. * batch_idx / len(val_loader)),
                          loss.item(), R.item(), r2.item()))
        if plot_res: 
            self.plot_results(y, out, self.num_targets)

    def mean_arr(self, num_epochs, arr):
        num_iter = int(len(arr) / num_epochs)
        mean_train_arr = [np.mean(arr[i*num_iter:(i+1)*num_iter]) for i in range(num_epochs)]
        return mean_train_arr
            
    def plot_metrics(self, num_epochs, train_arrs, val_arrs): 
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
        for i in range(3):
            mean_train_arr = self.mean_arr(num_epochs, train_arrs[i])
            mean_val_arr = self.mean_arr(num_epochs, val_arrs[i])
            axs[i].plot(np.arange(num_epochs), mean_train_arr, label='Train')
            axs[i].plot(np.arange(num_epochs), mean_val_arr, label='Val')
        fig.tight_layout()
        plt.show()    
    
    def calc_R_R2(self, y_true, y_pred, num_targets, device='cuda:0'):
        product = torch.sum(torch.multiply(y_true, y_pred), dim=1)
        true_sum = torch.sum(y_true, dim=1)
        true_sumsq = torch.sum(torch.square(y_true), dim=1)
        pred_sum = torch.sum(y_pred, dim=1)
        pred_sumsq = torch.sum(torch.square(y_pred), dim=1)
        count = torch.sum(torch.ones(y_true.shape), dim=1).to(device)
        true_mean = torch.divide(true_sum, count)
        true_mean2 = torch.square(true_mean)

        pred_mean = torch.divide(pred_sum, count)
        pred_mean2 = torch.square(pred_mean)

        term1 = product
        term2 = -torch.multiply(true_mean, pred_sum)
        term3 = -torch.multiply(pred_mean, true_sum)
        term4 = torch.multiply(count, torch.multiply(true_mean, pred_mean))
        covariance = term1 + term2 + term3 + term4

        true_var = true_sumsq - torch.multiply(count, true_mean2)
        pred_var = pred_sumsq - torch.multiply(count, pred_mean2)
        pred_var = torch.where(torch.greater(pred_var, 1e-12), pred_var, np.inf*torch.ones(pred_var.shape).to(device))

        tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))

        correlation = torch.divide(covariance, tp_var)
        correlation = correlation[~torch.isnan(correlation)]
        correlation_mean = torch.mean(correlation)
        total = torch.subtract(true_sumsq, torch.multiply(count, true_mean2))
        resid1 = pred_sumsq
        resid2 = -2*product 
        resid3 = true_sumsq
        resid = resid1 + resid2 + resid3 
        r2 = torch.ones_like(torch.tensor(num_targets)) - torch.divide(resid, total)
        r2 = r2[~torch.isinf(r2)]
        r2_mean = torch.mean(r2)
        return correlation_mean, r2_mean


        
    def regularize_loss(self, lambda1, ltype, net, loss):
        if ltype == 3:
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_1.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_2.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_3.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                    net.conv_block_4.parameters(), lambda1)
                torch.nn.utils.clip_grad_norm_(
                        net.conv_block_5.parameters(), lambda1)
                for i in range(len(net.dilations)):
                    torch.nn.utils.clip_grad_norm_(
                        net.dilations[i].parameters(), lambda1)

        else:      
            l0_params = torch.cat(
                [x.view(-1) for x in net.conv_block_1[1].parameters()])
            l1_params = torch.cat(
                [x.view(-1) for x in net.conv_block_2[1].parameters()])
            l2_params = torch.cat(
                [x.view(-1) for x in net.conv_block_3[1].parameters()])
            l3_params = torch.cat(
                [x.view(-1) for x in net.conv_block_4[1].parameters()])
            l4_params = torch.cat(
                    [x.view(-1) for x in net.conv_block_5[1].parameters()])
            dil_params = []
            for i in range(len(net.dilations)):
                dil_params.append(torch.cat(
                    [x.view(-1) for x in net.dilations[i][1].parameters()]))

        if ltype in [1, 2]:
            l1_l0 = lambda1 * torch.norm(l0_params, ltype)
            l1_l1 = lambda1 * torch.norm(l1_params, ltype)
            l1_l2 = lambda1 * torch.norm(l2_params, ltype)
            l1_l3 = lambda1 * torch.norm(l3_params, ltype)
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l1_l4 = lambda1 * torch.norm(l4_params, 2)
            dil_norm = []
            for d in dil_params:
                dil_norm.append(lambda1 * torch.norm(d, ltype))  
            loss = loss + l1_l0 + l1_l1 + l1_l2 + l1_l3 + l1_l4 + torch.stack(dil_norm).sum()

        elif ltype == 4:
            l1_l0 = lambda1 * torch.norm(l0_params, 1)
            l1_l1 = lambda1 * torch.norm(l1_params, 1)
            l1_l2 = lambda1 * torch.norm(l2_params, 1)
            l1_l3 = lambda1 * torch.norm(l3_params, 1)
            l2_l0 = lambda1 * torch.norm(l0_params, 2)
            l2_l1 = lambda1 * torch.norm(l1_params, 2)
            l2_l2 = lambda1 * torch.norm(l2_params, 2)
            l2_l3 = lambda1 * torch.norm(l3_params, 2)
            l1_l4 = lambda1 * torch.norm(l4_params, 1)
            l2_l4 = lambda1 * torch.norm(l4_params, 2)
            dil_norm1, dil_norm2 = [], []
            for d in dil_params:
                dil_norm1.append(lambda1 * torch.norm(d, 1))  
                dil_norm2.append(lambda1 * torch.norm(d, 2))  

            loss = loss + l1_l0 + l1_l1 + l1_l2 +\
                    l1_l3 + l1_l4 + l2_l0 + l2_l1 +\
                    l2_l2 + l2_l3 + l2_l4 + \
                torch.stack(dil_norm1).sum() + torch.stack(dil_norm2).sum()
        return loss

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

    def load_model(self, model, filename):
        model.load_state_dict(torch.load(filename))    
from __future__ import division
import numpy as np
import torch
import os
import logging
from torch.utils.data import DataLoader, Dataset, Sampler


class TrainDataset(Dataset):
    def __init__(self, X_train, X_train_trend, X_train_seasonal, X_train_resid,y_train,index = 'none'):
        if index != 'none':
            self.data = X_train[index]
            self.data_trend = X_train_trend[index]
            self.data_seasonal = X_train_seasonal[index]
            self.data_resid = X_train_resid[index]
            self.label = y_train[index]
        else:
            self.data = X_train
            self.data_trend = X_train_trend
            self.data_seasonal = X_train_seasonal
            self.data_resid = X_train_resid
            self.label = y_train
        self.train_len = self.data.shape[0]
    def __len__(self):
        return self.train_len

    def __getitem__(self, index):
        return index,self.data[index],self.data_trend[index],self.data_seasonal[index],self.data_resid[index],self.label[index]


class ValDataset(Dataset):
    def __init__(self, X_val,y_val):
        self.data = X_val
        self.label = y_val
        self.val_len = self.data.shape[0]

    def __len__(self):
        return self.val_len

    def __getitem__(self, index):
        return index,self.data[index],self.label[index]

class TestDataset(Dataset):
    def __init__(self, X_test,y_test):
        self.data = X_test
        self.label = y_test
        self.test_len = self.data.shape[0]

    def __len__(self):
        return self.test_len

    def __getitem__(self, index):
        return index,self.data[index],self.label[index]



def input_dataset(X_train, X_train_trend, X_train_seasonal, X_train_resid, X_val, y_train,y_val):
    train_set = TrainDataset(X_train, X_train_trend, X_train_seasonal, X_train_resid,y_train)
    val_set = ValDataset(X_val,y_val)
    return train_set,val_set





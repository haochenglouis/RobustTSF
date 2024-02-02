from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np
import random
import sys
from tqdm import trange
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL 
from trendfilter_modify import trend_filter, get_example_data,plot_model
from math import sqrt
from pandas import read_csv, DataFrame
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
from basemodel import LSTM,LSTM_v2
import torch
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(0)

####### modify the data path here ##############
ele_path = 'electricity/electricity.csv' 
traffic_path = 'traffic/traffic.csv' 
####### modify the data path here ##############

def adding_noise(X, noise_rate, noise_scale,noise_type = 'const'):
    random.seed(0)
    np.random.seed(0)
    noisy_X = np.zeros(len(X))
    #X_std = np.std(X)
    noise_r = noise_rate
    for i in range(len(X)):
        xi = X[i]
        if noise_type == 'const':
            noise_signal = noise_scale 
        elif noise_type == 'missing':
            noise_signal = -xi + 0.0001
        elif noise_type == 'gaussian':
            noise_signal = np.random.normal(loc=0,scale=noise_scale) 
        elif noise_type == 'none':
            noise_signal = 0
        elif noise_type == 'seq':
            if i==0:
                noise_signal = 0
            else:
                x_pre = noisy_X[i-1]
                true_pre = X[i-1]
                if x_pre==true_pre:
                    noise_r = 0.3
                    #noise_signal = noise_scale
                    #noise_signal = -xi + 0.0001
                    noise_signal = np.random.normal(loc=0,scale=noise_scale)
                else:
                    noise_r = noise_rate#noise_rate
                    noise_signal = -xi + x_pre + np.random.normal(loc=0,scale=0.01) 
        k = random.random()
        if k <=noise_r:
            noisy_X[i] = xi + noise_signal
        else:
            noisy_X[i] = xi
    return noisy_X

def adding_noise_select(X, total_sequence_length, noise_rate, noise_scale,noise_type = 'const',selection='Y'):
    random.seed(0)
    np.random.seed(0)
    num_X, length_X = X.shape[0], X.shape[1]
    noisy_X = np.zeros((num_X,length_X))
    all_positions = np.arange(total_sequence_length)
    if selection == 'Y':
        noise_position = [all_positions[-1]]
    elif selection == 'front':
        s_point = int(len(all_positions)*0.333)
        noise_position = list(all_positions[:s_point])
        #noise_position = list(all_positions[:s_point]) + [all_positions[-1]]
    elif selection == 'middle':
        s_point = int(len(all_positions)*0.333)
        ss_point = int(len(all_positions)*0.666)
        noise_position = list(all_positions[s_point:ss_point])
        #noise_position = list(all_positions[s_point:ss_point]) + [all_positions[-1]]
    elif selection == 'back':
        ss_point = int(len(all_positions)*0.666)
        #noise_position = list(all_positions[ss_point:])
        noise_position = list(all_positions[ss_point:-1])
    #noise_position = [0,1,2,3,4,5,6,7,8,9,10,16]
    for i in range(num_X):
        xi = X[i,:]
        add_noise = np.zeros(length_X)
        if noise_type == 'const':
            noise_signal = [noise_scale] * length_X
        elif noise_type == 'missing':
            noise_signal = -xi + 0.0001
        elif noise_type == 'gaussian':
            noise_signal = np.random.normal(loc=0,scale=noise_scale,size = length_X)
        elif noise_type == 'none':
            noise_signal = [0] * length_X
        for ii in noise_position:
            k = random.random()
            if k <=noise_rate:
                add_noise[ii] = noise_signal[ii]
        xi_noisy = xi + add_noise
        noisy_X[i,:] = xi_noisy
    return noisy_X

def data_to_2d(X,total_sequence_length,stride):
    daily_data = []
    for i in range(0, X.shape[0] - total_sequence_length, stride):
        daily_data.append(X[i:i+total_sequence_length])
    return np.stack(daily_data)

def create_X_data(dataset, time_step=1):
    dataX = []
    for i in range(len(dataset)):
        X_data = dataset[i][0:time_step]
        dataX.append(X_data)
    return np.array(dataX)

def create_y_data(dataset,train_time_steps):
    dataY = []
    dataY_binary = []
    for i in range(len(dataset)):
        y_data = max(dataset[i][train_time_steps:])
        dataY.append(y_data)
    return np.array(dataY)
def ready_X_data(train_data, train_data_trend,train_data_seasonal,train_data_resid,val_data,  train_time_steps):
    X_train = create_X_data(train_data, train_time_steps)
    X_train_trend = create_X_data(train_data_trend, train_time_steps)
    X_train_seasonal = create_X_data(train_data_seasonal, train_time_steps)
    X_train_resid = create_X_data(train_data_resid, train_time_steps)
    X_val = create_X_data(val_data, train_time_steps)
    # reshape input to be [samples, time steps, features] which is required for LSTM
    return X_train, X_train_trend, X_train_seasonal, X_train_resid, X_val

def ready_y_data(train_data, val_data,  train_time_steps):
    y_train = create_y_data(train_data, train_time_steps)
    y_val = create_y_data(val_data, train_time_steps)
    return y_train,y_val

def input_data(args,data_name = 'ele', ano_ratio = 0.2, ano_scale = 0.5, ano_type='const',selection='none',decompose = 'trendfilter',trendfilter_loss = 'mse',impute='none'):
    np.random.seed(0)
    if data_name == 'sin':
        points_per_cycle = 100
        #points_per_cycle = 30
        cycles = 100
        T = np.arange(0,cycles*2*np.pi,cycles*2*np.pi/(cycles*points_per_cycle))
        data = np.sin(T)
        stride = 1
        train_sequence_length = 16
        test_sequence_length = 1
        total_sequence_length = train_sequence_length + test_sequence_length
        train_ratio = 0.7
        val_ratio = 0.3
    elif data_name == 'ele':
        data_path = ele_path
        df = pd.read_csv(data_path,index_col=0)
        df = df['OT'].values
        data = np.array(df).reshape(-1)
        stride = 1
        train_sequence_length = 16
        test_sequence_length = 1
        total_sequence_length = train_sequence_length + test_sequence_length
        train_ratio = 0.7#0.8
        val_ratio = 0.3
    elif data_name == 'traffic':
        data_path = traffic_path
        df = pd.read_csv(data_path,index_col=0)
        df = df['OT'].values
        data = np.array(df).reshape(-1)
        stride = 1
        train_sequence_length = 16
        test_sequence_length = 1
        total_sequence_length = train_sequence_length + test_sequence_length
        train_ratio = 0.7#0.8
        val_ratio = 0.3#0.19
    train_end = int(len(data)*train_ratio)
    val_end = int(len(data)*(train_ratio+val_ratio))
    train_data = data[:train_end]
    val_data = data[train_end:val_end]

    
    scaler=StandardScaler()
    train_data=scaler.fit_transform(train_data.reshape(-1,1)).reshape(-1)
    
    
    #print('val persistence result:', np.mean(np.abs(np.diff(val_data))))
    if selection == 'none':
        train_data = adding_noise(train_data,noise_rate=ano_ratio,noise_scale=ano_scale,noise_type = ano_type)
        if decompose == 'trendfilter':
            trendfilter_result  = trend_filter(np.arange(len(train_data))/10,train_data,loss=trendfilter_loss,l_norm=1,alpha_2=0.3)
            train_data_trend = trendfilter_result['y_fit'].reshape(-1)
            train_data_seasonal = train_data_trend #position no use
            train_data_resid = train_data_trend    #position no use
            if impute != 'none':
                if impute == 'model_impute':
                    model = LSTM(args.n_features,args.seq_length,batch_size=args.batch_size, n_hidden=args.hidden_size, n_layers=args.lstm_layers)
                    model.load_state_dict(torch.load('results/' + args.dataset +'/base_lstm/'+str(args.ano_scale)+'_'+str(args.ano_ratio) +'_' +args.ano_type  +'_'+'best_model.pth')['state_dict'])
                    model.eval()
                    train_data_new = train_data.copy()    
                    hidden = torch.zeros(args.lstm_layers, 1,args.hidden_size).to(args.device)
                    cell = torch.zeros(args.lstm_layers, 1,args.hidden_size).to(args.device)  
                    sum_dect = 0
                    for pos in range(len(train_data_new)):
                        if pos > args.seq_length:
                            data_rev = torch.tensor(train_data_new[pos-args.seq_length:pos]).to(torch.float).reshape(1,args.seq_length,1)
                            pred_value,hidden,cell = model(data_rev,hidden,cell)
                            pred_value = float(torch.squeeze(pred_value))
                            if np.abs(pred_value-train_data[pos])>=args.model_dect_thre:
                                sum_dect+=1
                                train_data_new[pos] = pred_value
                    print('dect_ratio is',sum_dect/float(len(train_data)))
                    train_data = train_data_new
                if impute == 'online':
                    model = LSTM(args.n_features,args.seq_length,batch_size=args.batch_size, n_hidden=args.hidden_size, n_layers=args.lstm_layers)
                    model.load_state_dict(torch.load('results/' + args.dataset +'/base_lstm/'+str(args.ano_scale)+'_'+str(args.ano_ratio) +'_' +args.ano_type  +'_'+'best_model.pth')['state_dict'])
                    model.eval()
                    train_data_new = train_data.copy()    
                    hidden = torch.zeros(args.lstm_layers, 1,args.hidden_size).to(args.device)
                    cell = torch.zeros(args.lstm_layers, 1,args.hidden_size).to(args.device)  
                    sum_dect = 0
                    for pos in range(len(train_data_new)):
                        if pos > args.seq_length:
                            data_rev = torch.tensor(train_data_new[pos-args.seq_length:pos]).to(torch.float).reshape(1,args.seq_length,1)
                            pred_value,hidden,cell = model(data_rev,hidden,cell)
                            pred_value = float(torch.squeeze(pred_value))
                            if np.abs(pred_value-train_data[pos])>=args.model_dect_thre:
                                sum_dect+=1
                                train_data_new[pos] = 0
                    print('dect_ratio is',sum_dect/float(len(train_data)))
                    train_data = train_data_new
        elif (decompose == 'STL')|(decompose == 'STL_resid'):
            train_pd = pd.DataFrame(train_data,columns=['values'],index=pd.date_range('2021-01-01',periods=len(train_data),freq='D'))
            stl = STL(train_pd['values'])
            decompose = stl.fit()
            train_data_trend = np.array(decompose.trend)
            train_data_seasonal = np.array(decompose.seasonal)
            train_data_resid = np.array(decompose.resid)
        elif decompose == 'diff':
            train_data_length = len(train_data)
            train_data_trend = np.zeros(len(train_data_length))#position no use
            train_data_seasonal = np.zeros(len(train_data_length))#position no use
            train_data_resid = np.zeros(len(train_data_length))#position no use
        train_data = data_to_2d(train_data,total_sequence_length,stride)
    else:
        train_data_length = len(train_data)
        train_data = data_to_2d(train_data,total_sequence_length,stride)
        train_data = adding_noise_select(train_data,total_sequence_length,noise_rate=ano_ratio,noise_scale=ano_scale,noise_type = ano_type,selection=selection)
        train_data_trend = np.zeros(train_data_length)
        train_data_seasonal = np.zeros(train_data_length)
        train_data_resid = np.zeros(train_data_length)

    val_data=scaler.transform(val_data.reshape(-1,1)).reshape(-1)
    val_data = data_to_2d(val_data,total_sequence_length,stride)
    train_data_trend = data_to_2d(train_data_trend,total_sequence_length,stride)
    train_data_seasonal = train_data_trend
    train_data_resid = train_data_trend
    print('shape of train data',train_data.shape)
    print('shape of val data',val_data.shape)

    X_train, X_train_trend, X_train_seasonal, X_train_resid, X_val = ready_X_data(train_data, train_data_trend,train_data_seasonal,train_data_resid,val_data, train_sequence_length)
    y_train,y_val = ready_y_data(train_data, val_data, train_sequence_length)
    return train_sequence_length,X_train, X_train_trend, X_train_seasonal, X_train_resid, X_val,y_train,y_val






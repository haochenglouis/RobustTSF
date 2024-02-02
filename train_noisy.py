import argparse
import logging
import os
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from basemodel import LSTM
from loss import *
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from dataset import input_data
from dataloader import input_dataset
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ele', help='Name of the dataset [sin,ele,traffic]')
parser.add_argument('--epochs', type=int, default=30, help='Number of sweeps over the dataset to train')
parser.add_argument('--eval_iter', type=int, default=10, help='Number of iterations for evaluation')
parser.add_argument('--data_folder', default='data', help='Parent dir of the dataset')
parser.add_argument('--loss', default='mae', help='selected loss function')
parser.add_argument('--lstm_layers', type=int, default=2,help='number of layers in lstm')
parser.add_argument('--hidden_size', type=int, default=10,help='hidden dimension in lstm')
parser.add_argument('--n_features', type=int, default=1,help='input feature size in lstm')
parser.add_argument('--num_workers', type=int, default=1,help='number of workers')

parser.add_argument('--ano_ratio', type=float, default=0.3,help='ratio of anomalies')
parser.add_argument('--ano_scale', type=float, default=0.5,help='scale of anomalies')
parser.add_argument('--ano_type', type=str, default='const',help='type of anomalies [none,const,missing,gaussian]')
parser.add_argument('--selection', type = str, help = 'none, Y, front, middle or back',default = 'none')
parser.add_argument('--model_dect_thre', type = float, default=0.5,help = 'threshold for detecting anomaly')

parser.add_argument('--seq_length', type=int, default=16,help='sequence_length size in lstm')
parser.add_argument('--lr', type = float, default = 0.01)
parser.add_argument('--alpha', type = float, default = 1.0)
parser.add_argument('--beta', type = float, default = 0.0)
parser.add_argument('--thre', type = float, default = 0.3)
parser.add_argument('--anoscore', type = str, help = 'trendfilter, diff, or STL',default = 'trendfilter')
parser.add_argument('--win_size', type = int, help = 'for diff method',default = 1)
parser.add_argument('--trendfilter_loss', type = str, default = 'mae')
parser.add_argument('--model_type', type = str, default = 'LSTM')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples in each mini-batch')

parser.add_argument('--impute', type=str, default='none', help='[none,model_impute]')



args = parser.parse_args()

if torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if args.ano_type == 'gaussian':
    args.ano_scale = 2.0

if args.impute == 'none':
    args.this_method = 'base_lstm'
elif args.impute == 'model_impute':
    args.this_method = 'base_lstm_model_impute'
    
save_checkpoint_dir = 'results/' + args.dataset +'/' + args.this_method 
if not os.path.exists(save_checkpoint_dir):
    os.system('mkdir -p %s' % save_checkpoint_dir)


train_sequence_length,X_train, X_train_trend, X_train_seasonal, X_train_resid, \
X_val, y_train,y_val = input_data(args,data_name = args.dataset, ano_ratio = args.ano_ratio, \
                                    ano_scale = args.ano_scale, ano_type=args.ano_type,\
                                    selection=args.selection,decompose = args.anoscore,\
                                    trendfilter_loss = args.trendfilter_loss,\
                                    impute = args.impute)

args.seq_length = train_sequence_length

train_set,val_set = input_dataset(X_train, X_train_trend, X_train_seasonal, X_train_resid, \
X_val, y_train,y_val)


train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

if args.model_type == 'LSTM':
    model = LSTM(args.n_features,args.seq_length,batch_size=args.batch_size, n_hidden=args.hidden_size, n_layers=args.lstm_layers).to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loss_dict = {'cross_entropy':cross_entropy,'mse':mse,'rmse':rmse,'mae':mae,'evt_cross_entropy':evt_cross_entropy,'meta_ce':cross_entropy}
reg_criterion = loss_dict[args.loss]

best_mae = [10000]
best_mse = [10000]


def validate(loader, model):
    
    model.eval()
    total = 0
    pred_all = []
    mse_all = 0
    mae_all = 0
    with torch.no_grad():
        for ii, (index,samples,labels) in enumerate(loader):
            batch_size = samples.shape[0]
            samples = samples.reshape(samples.shape[0],samples.shape[1],1)
            samples = samples.to(torch.float32).to(args.device)
            labels = labels.to(torch.float32).to(args.device)
            hidden = torch.zeros(args.lstm_layers, batch_size, args.hidden_size).to(args.device)
            cell = torch.zeros(args.lstm_layers, batch_size, args.hidden_size).to(args.device)
            points, hidden,cell = model(samples,hidden,cell)
            mse_all += batch_size * F.mse_loss(torch.squeeze(points),labels)
            mae_all += batch_size * F.l1_loss(torch.squeeze(points),labels)
            total += batch_size
    mse_mean = (mse_all/float(total)).item()
    mae_mean = (mae_all/float(total)).item()
    return mse_mean,mae_mean

alpha_plan = [0.01]*10 + [0.001]*40

def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]


if __name__ == '__main__':
    for epoch in range(args.epochs):
        print('current epoch',epoch)
        adjust_learning_rate(optimizer, epoch, alpha_plan)
        model.train()
        #adjust_learning_rate(optimizer, epoch, alpha_plan)
        correct = 0
        total = 0
        mse_train_all = 0
        mae_train_all = 0
        for ii, (index,samples, samples_trend,samples_seasonal,samples_resid, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = samples.shape[0]
            samples = samples.reshape(samples.shape[0],samples.shape[1],1)
            samples = samples.to(torch.float32).to(args.device)
            labels = labels.to(torch.float32).to(args.device)
            hidden = torch.zeros(args.lstm_layers, batch_size, args.hidden_size).to(args.device)
            cell = torch.zeros(args.lstm_layers, batch_size, args.hidden_size).to(args.device)
            points,hidden,cell = model(samples,hidden,cell) 
            loss = reg_criterion(epoch,torch.squeeze(points),labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mse_train_all += batch_size * F.mse_loss(torch.squeeze(points),labels)
            mae_train_all += batch_size * F.l1_loss(torch.squeeze(points),labels)
            total += batch_size
        mse_train_mean = (mse_train_all/float(total)).item()
        mae_train_mean = (mae_train_all/float(total)).item()

        model.eval()
        mse_val,mae_val = validate(val_loader, model)
        if mae_val < best_mae[0]:
            best_mae[0] = mae_val
            best_mse[0] = mse_val
            torch.save({'state_dict': model.state_dict()},save_checkpoint_dir + '/' +str(args.ano_scale)+'_'+str(args.ano_ratio) +'_' +args.ano_type  +'_'+ 'best_model.pth')
        print('train mse is', mse_train_mean)
        print('train mae is', mae_train_mean)
        print('val mse (last) is', mse_val)
        print('val mae (last) is', mae_val)
        print('val mse (best) is', best_mse[0])
        print('val mae (best) is', best_mae[0])
        if epoch == args.epochs-1:
            torch.save({'state_dict': model.state_dict()},save_checkpoint_dir + '/' +str(args.ano_scale)+'_'+str(args.ano_ratio) +'_' +args.ano_type  +'_'+ 'last_model.pth')
   


    



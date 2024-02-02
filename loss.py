import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

def mse(epoch,logits,label,reduction=True):
    if reduction:
        return F.mse_loss(logits,label)
    else:
        return F.mse_loss(logits,label,reduction='none')


def rmse(epoch,logits,label,reduction=True):
    if reduction:
        return torch.sqrt(F.mse_loss(logits,label))
    else:
        return F.mse_loss(logits,label,reduction='none') ## sqrt later in main code


def mae(epoch,logits,label,reduction=True):
    if reduction:
        return F.l1_loss(logits,label)
    else:
        return F.l1_loss(logits,label,reduction='none')

def cross_entropy(epoch,logits,label,reduction=True):
    if reduction:
        return F.cross_entropy(logits, label)
    else:
        return F.cross_entropy(logits, label, reduction='none')


def evt_cross_entropy(epoch,logits,label,fq_normal,reduction=True):
    # using gamma=1
    num_batch = len(label)
    p = F.softmax(logits,dim=1)
    p_for_weighting = p[torch.arange(num_batch),label]
    weights = fq_normal*(1-p_for_weighting)*label + (1-fq_normal)*p_for_weighting * (1-label)
    loss = F.cross_entropy(logits, label, reduction = 'none') * weights
    if reduction:
        return torch.mean(loss)
    else:
        return loss


def loss_fn(mu, sigma, labels):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    '''
    zero_index = (labels != 0)
    distribution = torch.distributions.normal.Normal(mu[zero_index], sigma[zero_index])
    likelihood = distribution.log_prob(labels[zero_index])
    return -torch.mean(likelihood)







import os
import torch
import math
import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import metrics

from networks import *

def model_architecture(target_name, in_channel, cls_num, dropout = 0.0):
    return model_type(target_name, in_channel, cls_num, dropout)

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def check_dirs(name):
    folders = [
        'target_model/',
        'shadow_model/',
        'attack_model/',
        'log/',
        'result/'
    ]
    _folders = []
    for f in folders:
        _d = '%s/%s' % (name, f)
        check_dir(_d)
        _folders.append(_d)
    return _folders

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def categorial_cross_entropy_torch(outputs, targets): # numpy, numpy
    # as torch.nn.CrossEntropyLoss mean
    _loss = []
    for _id in range(outputs.shape[0]):
        mu = sum([np.exp(outputs[_id][i]) for i in range(outputs.shape[1])])
        _l = -np.log(np.exp(outputs[_id][targets[_id]]) / mu)
        _loss.append(_l)
    return _loss
    
def compare_loss_sigmoid(t_loss, c_loss):
    res = [_sigmoid(c_loss[i] - t_loss[i]) for i in range(len(c_loss))]
    return res

def loss_strategy(t_loss, c_loss, ty='sigmoid', th=0.5):
    return compare_loss_sigmoid(t_loss, c_loss)

def read_file(p):
    res = []
    with open(p, 'r') as f:
        lines = f.readlines()
        for line in lines:
            res.append(round(float(line.strip()), 6))
    return res

def compute_refer_loss(dir, models):
    loss = []
    k = 0
    for m in models:
        _loss = read_file(os.path.join(dir, m))
        if k == 0:
            loss = _loss
        else:
            for j in range(len(loss)):
                loss[j] += _loss[j]
        k = k + 1
    k = k * 1.0
    for j in range(len(loss)):
        loss[j] = loss[j] / k
    
    return loss

def manual_seed(seed):
    print("Setting seeds to: ", seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
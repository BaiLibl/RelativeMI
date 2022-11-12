import os
import random
import pandas as pd
import argparse
from datetime import datetime
from scipy.stats import pearsonr

from utils import *

def rank_accuracy(x, y):
    correct = 0
    total = 0
    for i in range(len(x)):
        for j in range(i):
            if (x[i] - x[j]) * (y[i] - y[j]) > 0:
                correct = correct + 1
            total += 1

    acc = 1.0 * correct / total
    return round(acc,6)

# ==========================================================================================================
# iteration-based method: select highest risk model as the reference model
# ==========================================================================================================

def privacy_risk_iteration(dir, K=10, target='cnn'):

    name, seed = dir.split('_')
    mtype = target.split('_')
    target_models = []
    for m in mtype:
        d = '%s_%s_%s/loss' % (name, m, seed)
        mo = os.listdir(d)
        target_models.extend(['%s/%s' % (d, i) for i in mo])
    
    iter_num = 0
    model_risk = {}
    served = []
    while iter_num < K:
        last_iter = {k:v for k,v in model_risk.items()}
        if len(model_risk) == 0:
            refer_model = random.choices(target_models, k=1)[0]
        else:
            sort_models = sorted(model_risk.items(), key = lambda kv:(kv[1], kv[0]))
            n = len(sort_models)
            for i in range(n):
                m = sort_models[n-i-1][0]
                if m not in served:
                    refer_model = m # highest risk
                    break

        served.append(refer_model)
        for cm in target_models:
            refer_model_path = refer_model
            candi_model_path = cm

            target_loss = read_file(candi_model_path)
            refer_loss = read_file(refer_model_path)
            risk = loss_strategy(target_loss, refer_loss, 'sigmoid')
            risk = [x for x in risk if str(x) != 'nan'] # exception
            diff = sum(risk) * 1.0 / len(risk)
            model_risk[cm] = diff
        print(iter_num, refer_model)
        # stop
        violates = [1 if model_risk[cm] > 0.5 else 0 for cm in target_models]
        if sum(violates) == 0:
            break

        iter_num = iter_num + 1
    
    return model_risk

# ==========================================================================================================
# experiment:
# 1. compare overfit with proposed method
# 2. compare random-based with iteration-based method
# ==========================================================================================================

def show_result(dir, key, K=1, _type='rand', ref=None, target='cnn'):

    name, seed = dir.split('_')
    mtype = target.split('_')
    print(123, mtype)
    _dir = '%s_%s_%s' % (name, mtype[0], seed)
    df = pd.read_csv(os.path.join(_dir, 'eval_%s.csv' % _dir))
    for i in range(len(mtype)-1):
        d = '%s_%s_%s' % (name, mtype[i+1], seed)
        df2 = pd.read_csv(os.path.join(d, 'eval_%s.csv' % d))
        df = pd.concat([df, df2], axis=0)
    
    oft = []
    att = []
    noise = []
    risk = []
    risks = privacy_risk_iteration(dir, 5, target)
    
    risks = {k.split('/')[-1]:risks[k] for k in risks.keys()}

    mds = list(df['model_name'])
    _target = []
    _network = []

    for m in mds:
        model_name = m
        if model_name not in risks.keys():
            continue

        oft.append(float(df[df['model_name']==m]['t_overfitting']))
        att.append(float(df[df['model_name']==m][key]))
        noise.append(float(model_name.split('_')[2]))

        risk.append(risks[model_name])
        c = df[df['model_name']==m]['model_name']
        _target.append(list(c)[0])
        c = df[df['model_name']==m]['target_name']
        _network.append(list(c)[0])
    
    oft_p = pearsonr(oft, att)[0]
    row = {
        'model_name':_target, 
        'target_name':_network, 
        't_overfitting':oft,
        'rmi':risk,
        'max_acc':att
    }
    
    df = pd.DataFrame(row)

    r1 = pearsonr(list(df['max_acc']), list(df['t_overfitting']))[0]
    r2 = pearsonr(list(df['max_acc']), list(df['rmi']))[0]
    print('PCC: overfit:%.4f, rmi:%.4f' % (r1, r2))
    r1 = rank_accuracy(list(df['max_acc']), list(df['t_overfitting']))
    r2 = rank_accuracy(list(df['max_acc']), list(df['rmi']))
    print('KRC: overfit:%.4f, rmi:%.4f' % (r1, r2))

    row = {
        'seed':seed, 
        'target_model':target, 
        'pcc':pearsonr(list(df['max_acc']), list(df['rmi']))[0],
        'krc':rank_accuracy(list(df['max_acc']), list(df['rmi']))
    }
    
    df = pd.DataFrame(row,index=[0])
    outfile='results/archres_%s.csv' % (dir.split('_')[0])
    if os.path.isfile(outfile):
        df.to_csv(outfile, mode='a', header=False, index=False)
    else:
        df.to_csv(outfile, mode='a', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--dir', type=str, default="stl10_1")
    parser.add_argument('-mm', '--mia_mt', type=str, default="max_acc") # sd_adv or max_acc
    parser.add_argument('-tp', '--type', type=str, default="iter") 
    parser.add_argument('-rf', '--refer', type=str, default=None) 
    parser.add_argument('-tg', '--target', type=str, default='cnn') 

    args = parser.parse_args()
    print(args)
    dir = args.dir
    mia_mt = args.mia_mt
    _type = args.type

    show_result(dir, mia_mt, _type=_type, target=args.target)

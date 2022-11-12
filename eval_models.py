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

def privacy_risk_iteration(dir):
    loss_dir = os.path.join(dir, 'loss/')
    iter_num = 0
    model_risk = {}
    candidate_models = os.listdir(loss_dir)
    model_name = [f for f in candidate_models]
    K = len(model_name)
    served = []
    while iter_num < K:
        last_iter = {k:v for k,v in model_risk.items()}
        if len(model_risk) == 0:
            refer_model = random.choices(model_name, k=1)[0]
        else:
            sort_models = sorted(model_risk.items(), key = lambda kv:(kv[1], kv[0]))
            n = len(sort_models)
            for i in range(n):
                m = sort_models[n-i-1][0]
                if m not in served:
                    refer_model = m # highest risk
                    break
        
        served.append(refer_model)
        for cm in candidate_models:
            refer_model_path = os.path.join(loss_dir, refer_model)
            candi_model_path = os.path.join(loss_dir, cm)

            target_loss = read_file(candi_model_path)
            refer_loss = read_file(refer_model_path)
            risk = loss_strategy(target_loss, refer_loss, 'sigmoid')
            risk = [x for x in risk if str(x) != 'nan'] # exception
            diff = sum(risk) * 1.0 / len(risk)
            model_risk[cm] = diff
        print("Iter %s: current reference:%s" % (iter_num, refer_model))
        # stop
        violates = [1 if model_risk[cm] > 0.5 else 0 for cm in candidate_models]
        if sum(violates) == 0:
            break

        iter_num = iter_num + 1
    
    return model_risk

# ==========================================================================================================
# random-based method: select a model randomly as the reference model
# ==========================================================================================================
def privacy_risk_random(dir, K=1, ref=None):
    loss_dir = os.path.join(dir, 'loss/')
    if ref == None:
        models = os.listdir(loss_dir)
        r_models = random.choices(models, k=K)
    else:
        r_models = [ref]
    print(r_models)
    refer_loss = compute_refer_loss(loss_dir, r_models)

    model_risk = {}
    for f in os.listdir(loss_dir):
        target_loss = read_file(os.path.join(loss_dir, f))
        risk = loss_strategy(target_loss, refer_loss, 'sigmoid')
        risk = [x for x in risk if str(x) != 'nan'] # maybe nan
        diff = sum(risk) * 1.0 / len(risk)
        model_risk[f] = diff

    return model_risk

# ==========================================================================================================
# experiment:
# 1. compare overfit with proposed method
# 2. compare random-based with iteration-based method
# ==========================================================================================================

def show_result(dir, key, K=1, _type='rand', ref=None):

    # df = pd.read_csv(os.path.join(dir, 'eval_%s.csv' % dir))
    df = pd.read_csv(os.path.join('results', 'eval_%s.csv' % dir))

    oft = []
    att = []
    noise = []
    risk = []
    if _type == 'rand':
        risks = privacy_risk_random(dir, K, ref)
    else:
        risks = privacy_risk_iteration(dir)
    
    for row in range(len(df['model_name'])):
        model_name = df['model_name'][row]
        if model_name not in risks.keys():
            continue

        oft.append(df['t_overfitting'][row])
        att.append(df[key][row])
        noise.append(float(model_name.split('_')[2]))

        risk.append(risks[model_name])
    
    oft_p = pearsonr(oft, att)[0]    
    oft_p = pearsonr(oft, att)[0]
    oft_r = rank_accuracy(oft, att)

    rsk_p = pearsonr(risk, att)[0]
    rsk_r = rank_accuracy(risk, att)

    print('Pearsonr: oft-%s:%.4f, rsk-%s:%.4f' % (key, oft_p, key,  rsk_p))
    print('Rank acc: oft-%s:%.4f, rsk-%s:%.4f' % (key, oft_r, key,  rsk_r))
    
    row = {
      'dataset': dir.split('_')[0],
      'target_name': dir.split('_')[1],
      'random_seed': dir.split('_')[2],
      'mia_mt': key,
      'type': _type,
      'oft_p': round(oft_p, 4),
      'oft_r': round(oft_r, 4),
      'rsk_p': round(rsk_p, 4),
      'rsk_r': round(rsk_r, 4),
      'time': datetime.now().strftime("%H%M") 
    }
    
    df = pd.DataFrame(row,index=[0])
    outfile='results/merge_%s.csv' % (dir.split('_')[0])
    if os.path.isfile(outfile):
        df.to_csv(outfile, mode='a', header=False, index=False)
    else:
        df.to_csv(outfile, mode='a', header=True, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--dir', type=str, default="location_mlp_0")
    parser.add_argument('-mm', '--mia_mt', type=str, default="sd_adv") # sd_adv or max_acc
    parser.add_argument('-tp', '--type', type=str, default="iter") # random or 
    parser.add_argument('-rf', '--refer', type=str, default=None) # random or 

    args = parser.parse_args()
    print(args)
    dir = args.dir
    mia_mt = args.mia_mt
    _type = args.type

    show_result(dir, mia_mt, _type=_type, ref=args.refer)

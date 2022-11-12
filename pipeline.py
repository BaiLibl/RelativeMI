import numpy as np
import argparse
import logging, os
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataloader import *
from networks import *
from config import *
from train import *
from MIA_metric import *
from MIA_shadow import *
from utils import check_dirs, check_dir, model_architecture, manual_seed

# model training & mount MIAs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--dataset', type=str, default="mnist")
    parser.add_argument('-tn', '--target_name', type=str, default="cnn")
    parser.add_argument('-ep', '--epoch', type=int, default=1)

    parser.add_argument('-dr', '--dropout', type=float, default=0.0)
    parser.add_argument('-l2', '--L2', type=float, default=0.0)
    parser.add_argument('-ne', '--noise', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()
    print(args)
    manual_seed(args.random_seed)
    name = args.dataset
    target_name = args.target_name
    epoch = args.epoch
    dr = args.dropout
    l2 = args.L2
    noise = args.noise
    use_DP = 1 if args.noise > 0 else 0
    shadow_model_path = 'shadow_model.pth'
    device = torch.device("cuda:%s" % (args.random_seed % GPU_NUM) if use_cuda else "cpu")

    _dir = '%s_%s_%s' % (name, target_name, args.random_seed)
    TARGET_MODEL_DIR, SHADOW_MODEL_DIR, ATTACK_MODEL_DIR, LOG_DIR, RESULT_DIR = check_dirs(_dir)
    current_time = datetime.now().strftime("%H%M") # %m%d%H%M
    model_name =  '%s_%s_%s_%s_%s_%s' % (name, target_name, noise, dr, l2, current_time)

    logging.basicConfig(filename='%s/%s.txt' % (LOG_DIR, model_name), datefmt='%Y-%m-%d %H:%M:%S', level = logging.DEBUG, format=LOG_FORMAT)
    logger = logging.getLogger('debug')
    writer = SummaryWriter(log_dir='%s/runs_target/%s' % (_dir, model_name))

    # ====================================================================================================
    # train target model
    # ====================================================================================================
    
    input_channel, cls_num, target_train, target_test, shadow_train, shadow_test = prepare_dataset(name, DATA_ROOT)
    target_model = model_architecture(target_name, input_channel, cls_num, dr)
    TARGET_PATH = '%s%s_target.pth' % (TARGET_MODEL_DIR, model_name)
    print('Target model:%s' % TARGET_PATH)
    t_acc_train, t_acc_test, t_overfitting = train_model(writer, logger, TARGET_PATH, target_model,  target_train, target_test\
                        ,use_DP, noise, l2, BATCH_SIZE, epoch, device)

    test_model(logger, TARGET_PATH, target_model, target_test)
    check_dir(_dir+'/loss/')
    filename = "%s/%s" % (_dir+'/loss/', model_name)
    get_loss(target_model, target_train, filename, device)
  
    # ====================================================================================================
    # train attack model
    # ====================================================================================================
    
    shadow_model = model_architecture(target_name, input_channel, cls_num, dr)
    SHADOW_PATH = '%s%s' % (SHADOW_MODEL_DIR, shadow_model_path)
    print('Shadow model:%s' % (SHADOW_PATH))

    target_model.load_state_dict(torch.load(TARGET_PATH))
    shadow_model.load_state_dict(torch.load(SHADOW_PATH))

    ATTACK_PATH = ATTACK_MODEL_DIR + model_name
    attack_model = ShadowAttackModel(cls_num)
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, 128)
    black_shadow(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, ATTACK_PATH, 100, device)

    # ====================================================================================================
    # shadow-training MIA
    # ====================================================================================================
    print('\n... Shadow-training MIA ...')
    mia_shadow = shadow_eval(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, ATTACK_PATH)
    acc_test, f1, auc, recall, precision, adv = mia_shadow[1:]
    mia_shadow = mia_shadow[1:]
    print('mia_shadow: acc:%.4f, R:%.4f, P:%.4f. f1:%.4f, auc:%.4f, adv:%.4f' % (acc_test, recall, precision, f1, auc, adv))

    # ====================================================================================================
    # threshod-setting MIA
    # ====================================================================================================
    print('\n... Threshod-setting  MIA ...')
    result2 = metric_eval(target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, 100, device)
    
    # ====================================================================================================
    # write to csv
    # ====================================================================================================

    logger.info(model_name)
    logger.info(args)
    logger.info('Target: t_acc_train:%.4f, t_acc_test:%.4f, t_overfitting:%.4f' % (t_acc_train, t_acc_test, t_overfitting))
    logger.info('shadow MIA: %s' % mia_shadow)
    logger.info('metric MIA: %s' % result2)

    row = {
        'random_seed': args.random_seed,
        'model_name': model_name, 
        'dataset':name,
        'target_name':target_name,
        'noise':noise,
        'weight_decay':l2,
        'dropout':dr,
        'epoch':epoch,
        'cur_time': current_time,
        't_acc_train':t_acc_train, 
        't_acc_test':t_acc_test,
        't_overfitting':t_overfitting
    }
    
    result2.append(['sd', mia_shadow])
    mia_mt = ['acc', 'f1', 'auc', 'R', 'P', 'adv']
    max_mt = [0.0 for i in range(len(mia_mt))]
    for item in result2:
        ty, v = item
        for i in range(len(v)):
            key = "%s_%s" % (ty, mia_mt[i])
            row[key] = v[i]
            max_mt[i] = max(v[i], max_mt[i])
    for i in range(len(mia_mt)):
        row['max_%s'%mia_mt[i]] = max_mt[i]

    df = pd.DataFrame(row,index=[0])
    outfile='%s/eval_%s_%s_%s.csv' % (_dir, name, target_name, args.random_seed)
    if os.path.isfile(outfile):
        df.to_csv(outfile, mode='a', header=False, index=False)
    else:
        df.to_csv(outfile, mode='a', header=True, index=False)

if __name__ == "__main__":
    main()
import numpy as np
import argparse
import logging
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from dataloader import *
from networks import *
from config import *
from train import *
from MIA_metric import *
from MIA_shadow import *
from utils import check_dirs, model_architecture, manual_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--dataset', type=str, default="mnist")
    parser.add_argument('-tn', '--target_name', type=str, default="cnn")
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()
    manual_seed(args.random_seed)
    name = args.dataset
    target_name = args.target_name
    dr = params[name]['shadow']['dropout']
    l2 = params[name]['shadow']['reg']
    epoch = params[name]['shadow']['epoch']
    noise = 0.0
    use_DP =  0

    device = torch.device("cuda:%s" % (args.random_seed % GPU_NUM) if use_cuda else "cpu")
    _dir = '%s_%s_%s' % (name, target_name, args.random_seed)
    _, SHADOW_DIR, _, LOG_DIR, _ = check_dirs(_dir)

    input_channel, cls_num, _, _, shadow_train, shadow_test = prepare_dataset(name, DATA_ROOT)
    shadow_model = model_architecture(target_name, input_channel, cls_num, dr)
    
    current_time = datetime.now().strftime("%H%M") # %m%d%H%M
    model_name =  'shadow_model'
    print('\n... Train %s ...' % model_name)

    logging.basicConfig(filename='%s/shadow_%s.txt' % (LOG_DIR, model_name), datefmt='%Y-%m-%d %H:%M:%S', level = logging.DEBUG, format=LOG_FORMAT)
    logger = logging.getLogger('debug')
    logger.debug(args)

    SHADOW_PATH = '%s%s.pth' % (SHADOW_DIR, model_name)
    writer = SummaryWriter(log_dir='%s/runs_shadow/%s' % (_dir, model_name))
    train_set, test_set = shadow_train, shadow_test
    t_acc_train, t_acc_test, t_overfitting = train_model(writer, logger, SHADOW_PATH, shadow_model, train_set, test_set\
                        ,use_DP, noise, l2, \
                        BATCH_SIZE, epoch, device)
    print('train_acc:%.4f, test_acc:%.4f, overfit:%.4f' % (t_acc_train, t_acc_test, t_overfitting))

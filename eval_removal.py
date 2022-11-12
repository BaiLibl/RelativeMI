import numpy as np
import argparse
import logging
import random
import pandas as pd
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


from dataloader import *
from networks import *
from config import *
from train import *
from MIA_metric import *
from MIA_shadow import *
from utils import check_dirs, model_architecture, manual_seed, check_dir


class SubsetDataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, exclude_indices=None, include_indices=None):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(SubsetDataWrapper, self).__init__()

        if exclude_indices is None:
            assert include_indices is not None
        if include_indices is None:
            assert exclude_indices is not None

        self.dataset = dataset

        if include_indices is not None:
            self.include_indices = include_indices
        else:
            S = set(exclude_indices)
            self.include_indices = [idx for idx in range(len(dataset)) if idx not in S]

        # record important attributes
        if hasattr(dataset, 'statistics'):
            self.statistics = dataset.statistics

    def __len__(self):
        return len(self.include_indices)

    def __getitem__(self, idx):
        real_idx = self.include_indices[idx]
        return self.dataset[real_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('-dl', '--delete', type=float, default=0.0)
    parser.add_argument('-db', '--dataset', type=str, default='purchase')
    parser.add_argument('-rk', '--risky', type=int, default=1)
    parser.add_argument('-tn', '--target_name', type=str, default='mlp')


    args = parser.parse_args()
    manual_seed(args.random_seed)
    name = args.dataset
    del_ratio = args.delete
    _dir = '%s_%s_%s' % (name, args.target_name, args.random_seed)
    shadow_model_path = 'shadow_model.pth'
    device = torch.device("cuda:%s" % (args.random_seed % GPU_NUM) if use_cuda else "cpu")

    print(args)

    ty = {
    0:'rd',
    1:'rk',
    2:'rw'
    }

    model_name = 'td_%s_%s_%s_%s.pth' % (name, args.random_seed, ty[args.risky], del_ratio)

    TARGET_MODEL_DIR, SHADOW_MODEL_DIR, ATTACK_MODEL_DIR, LOG_DIR, RESULT_DIR = check_dirs(_dir)
    current_time = datetime.now().strftime("%H%M") # %m%d%H%M

    logging.basicConfig(filename='%s/%s.txt' % (LOG_DIR, model_name), datefmt='%Y-%m-%d %H:%M:%S', level = logging.DEBUG, format=LOG_FORMAT)
    logger = logging.getLogger('debug')
    writer = SummaryWriter(log_dir='%s/runs_target/%s' % (_dir, model_name))

    # ====================================================================================================
    # train target model
    # ====================================================================================================
    dr = 0.0
    target_name = args.target_name
    use_DP = 0
    noise = 0
    l2 = 0
    epoch = 100
    
    input_channel, cls_num, target_train, target_test, shadow_train, shadow_test = prepare_dataset(name, DATA_ROOT)
    target_model = model_architecture(target_name, input_channel, cls_num, dr)
    check_dir('del_%s' % TARGET_MODEL_DIR)
    TARGET_PATH = 'del_%s%s' % (TARGET_MODEL_DIR, model_name)
    print('Target model:%s' % TARGET_PATH)

    n = len(target_train)
    n_del = int(n*float(args.delete))

    print('selection samples ...')
    loss_diff = []
    refer_model_1 = model_architecture(target_name, input_channel, cls_num)
    refer_model_2 = model_architecture(target_name, input_channel, cls_num)
    refer_model_1.load_state_dict(torch.load('%s/shadow_model/%s_refer_0.0.model.pth' % (_dir, name)))
    refer_model_2.load_state_dict(torch.load('%s/shadow_model/%s_refer_0.2.model.pth' % (_dir, name)))
    loss_function = nn.CrossEntropyLoss(reduction='none')

    refer_model_1 = refer_model_1.cuda()
    refer_model_2 = refer_model_2.cuda()
    trainloader = torch.utils.data.DataLoader(target_train, batch_size=128, shuffle=False, num_workers=2)
    for _, (inputs, targets) in enumerate(trainloader):
        if isinstance(targets, list):
            targets = targets[0]
          
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        outputs = refer_model_1(inputs)
        loss1 = loss_function(outputs, targets)

        outputs = refer_model_2(inputs)
        loss2 = loss_function(outputs, targets)

        dloss = loss1 - loss2
        loss_diff.append(dloss.data.cpu().numpy())

    final_outputs=np.concatenate(loss_diff)
    loss_diff_tensor=(torch.from_numpy(final_outputs).type(torch.FloatTensor))
    _, index = torch.sort(loss_diff_tensor, descending=False)
    print(index[:10], index[-10:])
    print(loss_diff_tensor[index[:10]], loss_diff_tensor[index[-10:]])

    if args.risky == 0:
        target_train_tmp, _ = torch.utils.data.random_split(target_train, [n-n_del, n_del])
    elif args.risky == 1:
        _index = index[:n-n_del]
        target_train_tmp = SubsetDataWrapper(target_train, include_indices=_index)
    elif args.risky == 2:
        _index = index[n_del:]
        target_train_tmp = SubsetDataWrapper(target_train, include_indices=_index)
    else:
        print('risky parameter', args.risky, 'error')

    print('n_del:%d train size:%d test size:%d' % (n_del, len(target_train_tmp), len(target_test)))

    # ====================================================================================================
    # train target model
    # ====================================================================================================
    print('train_tmp_size:%d' % len(target_train_tmp))
    t_acc_train, t_acc_test, t_overfitting = train_model(writer, logger, TARGET_PATH, target_model,  target_train_tmp, target_test\
                        ,use_DP, noise, l2, BATCH_SIZE, epoch, device)

    # ====================================================================================================
    # train attack model
    # ====================================================================================================
    
    shadow_model = model_architecture(target_name, input_channel, cls_num, dr)
    SHADOW_PATH = '%s%s' % (SHADOW_MODEL_DIR, shadow_model_path)

    target_model.load_state_dict(torch.load(TARGET_PATH))
    shadow_model.load_state_dict(torch.load(SHADOW_PATH))

    ATTACK_PATH = ATTACK_MODEL_DIR + model_name
    attack_model = ShadowAttackModel(cls_num)
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, 128)
    black_shadow(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, ATTACK_PATH, epoch, device)

    # ====================================================================================================
    # shadow-training MIA
    # ====================================================================================================
    
    mia_shadow = shadow_eval(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, ATTACK_PATH)
    mia_acc, f1, auc, recall, precision, adv = mia_shadow[1:]
    mia_shadow = mia_shadow[1:]
    print('mia_shadow: acc:%.4f, R:%.4f, P:%.4f. f1:%.4f, auc:%.4f, adv:%.4f' % (mia_acc, recall, precision, f1, auc, adv))

    # ====================================================================================================
    # threshod-setting MIA
    # ====================================================================================================
    result2 = metric_eval(target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, 100, device)

    metric = ['acc', 'f1', 'auc', 'R', 'P', 'adv']
    mth = ['corr', 'conf', 'ent', 'ment', 'sd']
    for item in result2:
        _, v = item
        mia_acc = max(mia_acc, v[0])
        adv = max(adv, v[5])

    tmp = {
    'type':args.risky,
    'delete':args.delete,
    'test_acc':round(t_acc_test,2),
    'train_acc':round(t_acc_train,2),
    'mia_acc':mia_acc,
    'mia_adv':adv,
    'time':datetime.now().strftime("%d-%H-%M")
    }
    outfile='results/remo_%s.csv' % (_dir)
    df = pd.DataFrame(tmp,index=[0])
    if os.path.isfile(outfile):
        df.to_csv(outfile, mode='a', header=False, index=False)
    else:
        df.to_csv(outfile, mode='a', header=True, index=False)
    

if __name__ == "__main__":
    main()
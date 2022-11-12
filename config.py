import torch

use_cuda = torch.cuda.is_available()
GPU_NUM=4
A_EPOCH = 1
BATCH_SIZE = 128
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
DATA_ROOT = '../data'

params = {
    'purchase':{
        'net':'mlp',
        'shadow':{
            'dropout':0.1,
            'reg':5e-4,
            'epoch':100
        }
    },
    'location':{
        'net':'mlp',
        'shadow':{
            'dropout':0.1,
            'reg':5e-4,
            'epoch':100
        }
    },
    'stl10':{
        'net':'cnn',
        'shadow':{
            'dropout':0.2,
            'reg':5e-4,
            'epoch':100
        }
    },
    'fmnist':{
        'net':'cnn',
        'shadow':{
            'dropout':0.2,
            'reg':5e-4,
            'epoch':100
        }
    },
    'cifar10':{ # stl10 is similar like stl10
        'net':'alex',
        'shadow':{
            'dropout':0.2,
            'reg':5e-4,
            'epoch':100
        }
    },
}
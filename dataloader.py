import os
import numpy as np
import urllib
import tarfile
import torch
import torchvision
import torchvision.transforms as transforms

import ssl
ssl._create_default_https_context = ssl._create_unverified_context # to download CIFAR10


def prepare_dataset(dataset_name, root):

    if dataset_name == 'purchase':
        cls_num, input_channel, target_train, target_test, shadow_train, shadow_test = prepare_purchase_dataset(root)
    elif dataset_name == 'location':
        cls_num, input_channel, target_train, target_test, shadow_train, shadow_test = prepare_location_dataset(root)
    else:
        input_channel, cls_num, train_set, test_set = get_model_dataset(dataset_name, root=root)
        dataset = train_set + test_set
        length = len(dataset)
        print('%s Dataset Size: %d' % (dataset_name, length))
        each_length = length//4
        target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    return input_channel, cls_num, target_train, target_test, shadow_train, shadow_test  

def get_model_dataset(dataset_name, root):
    if dataset_name.lower() == "cifar10":
        num_class = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = torchvision.datasets.CIFAR10(
                root=root, train=True, download=True, transform=transform)
            
        test_set = torchvision.datasets.CIFAR10(
               root=root, train=False, download=True, transform=transform)

        input_channel = 3
    elif dataset_name.lower() == "cifar100":
        num_class = 100
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize [-1,1]
        ])

        train_set = torchvision.datasets.CIFAR100(
                root=root, train=True, download=True, transform=transform)
            
        test_set = torchvision.datasets.CIFAR100(
               root=root, train=False, download=True, transform=transform)

        input_channel = 3
    elif dataset_name.lower() == "fmnist":
        num_class = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
                root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
                root=root, train=False, download=True, transform=transform)

        input_channel = 1
    
    elif dataset_name.lower() == "stl10":
        num_class = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.STL10(root=root, split='train', transform=transform, download=True)
        test_set = torchvision.datasets.STL10(root=root, split='test', transform=transform, download=True)
        input_channel = 3
        
    return input_channel, num_class, train_set, test_set

def tensor_data_create(features, labels):
    tensor_x = torch.stack([torch.FloatTensor(i) for i in features])
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

def prepare_purchase_dataset(root):
    cls_num = 100
    feature_num = 600
    DATASET_PATH = '%s/purchase' % root
    DATASET_IDX  = '%s/purchase/random_r_purchase100.npy' % root
    DATASET_NAME = 'dataset_purchase'
    DATASET_NUMPY = 'data.npz'

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)
    
    DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)
    
    if not os.path.isfile(DATASET_FILE):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')
        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)
    
        print('reading dataset...')
        data_set =np.genfromtxt(DATASET_FILE,delimiter=',')
        print('finish reading!')
        X = data_set[:,1:].astype(np.float64)
        Y = (data_set[:,0]).astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)
    
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']

    y_min = np.min(np.unique(Y))
    y_data = Y - y_min
    len_train =len(X)
    r = np.load(DATASET_IDX)

    X=X[r]
    Y=Y[r]
        
    train_classifier_ratio, train_attack_ratio = 0.1,0.3
    train_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    train_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len//2]
    target_indices = r[train_len//2:]

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    test_len = 1*train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len//2]
    target_indices = r[test_len//2:]
    
    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    target_train = tensor_data_create(target_train_data, target_train_label)
    target_test = tensor_data_create(target_test_data, target_test_label)
    return cls_num, feature_num, target_train, target_test, shadow_train, shadow_test

def prepare_texas_dataset(root):
    cls_num = 100
    feature_num = 6169
    DATASET_PATH = '%s/texas/' % root
    DATASET_IDX  = '%s/texas/random_r_texas100.npy' % root
    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    DATASET_FEATURES = os.path.join(DATASET_PATH,'texas/100/feats')
    DATASET_LABELS = os.path.join(DATASET_PATH,'texas/100/labels')
    DATASET_NUMPY = 'data.npz'
    
    if not os.path.isfile(DATASET_FEATURES):
        print('Dowloading the dataset...')
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')

        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)
        print('reading dataset...')
        data_set_features =np.genfromtxt(DATASET_FEATURES,delimiter=',')
        data_set_label =np.genfromtxt(DATASET_LABELS,delimiter=',')
        print('finish reading!')

        X =data_set_features.astype(np.float64)
        Y = data_set_label.astype(np.int32)-1
        np.savez(os.path.join(DATASET_PATH, DATASET_NUMPY), X=X, Y=Y)
    
    data = np.load(os.path.join(DATASET_PATH, DATASET_NUMPY))
    X = data['X']
    Y = data['Y']
    y_min=np.min(np.unique(Y))
    y_data = Y - y_min
    r = np.load(DATASET_IDX)
    X=X[r]
    Y=Y[r]

    len_train =len(X)
    train_classifier_ratio, train_attack_ratio = float(10000)/float(X.shape[0]),0.3
    train_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    train_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    np.random.seed(100)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len//2]
    target_indices = np.delete(np.arange(train_len), shadow_indices)

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]


    test_len = 1*train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len//2]
    target_indices = np.delete(np.arange(test_len), shadow_indices)

    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]


    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    target_train = tensor_data_create(target_train_data, target_train_label)
    target_test = tensor_data_create(target_test_data, target_test_label)
    return cls_num, feature_num, target_train, target_test, shadow_train, shadow_test

def prepare_location_dataset(root):
    cls_num = 30
    feature_num = 446
    DATASET_PATH = '%s/location/data_complete.npz' % root
    DATASET_IDX  = '%s/location/shuffle_index.npz' % root

    npzdata=np.load(DATASET_PATH)
    x_data=npzdata['x'][:,:]
    y_data=npzdata['y'][:]
    y_min=np.min(np.unique(y_data))
    y_data = y_data - y_min
    npzdata_index=np.load(DATASET_IDX)
    index_data=npzdata_index['x']
    np.random.shuffle(index_data)
    
    target_train_x, target_train_y = x_data[index_data[0:1000]], y_data[index_data[0:1000]]
    target_test_x, target_test_y = x_data[index_data[1000:2000]], y_data[index_data[1000:2000]]
    shadow_train_x, shadow_train_y = x_data[index_data[2000:3000]], y_data[index_data[2000:3000]]
    shadow_test_x,  shadow_test_y = x_data[index_data[3000:4000]], y_data[index_data[3000:4000]]


    target_train = tensor_data_create(target_train_x, target_train_y)
    target_test  = tensor_data_create(target_test_x, target_test_y)
    shadow_train  = tensor_data_create(shadow_train_x, shadow_train_y)
    shadow_test  = tensor_data_create(shadow_test_x, shadow_test_y)

    return cls_num, feature_num, target_train, target_test, shadow_train, shadow_test
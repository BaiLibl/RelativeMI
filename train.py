import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

np.set_printoptions(threshold=np.inf)

from opacus import PrivacyEngine # opacus requires Python version >=3.6.8 || DP uses
from torch.optim import lr_scheduler
from opacus.utils import module_modification
from opacus.dp_model_inspector import DPModelInspector


class model_training():
    def __init__(self, trainloader, testloader, model, use_DP = 0, noise=1.3, l2=5e-4, logger=None, device='cuda'): #l2=0
        grad_norm = 1.5
        self.use_DP = use_DP
        self.device = device
        self.net = model.to(self.device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.logger = logger

        # use all GPU
        # if self.device == 'cuda':
        #     self.net = torch.nn.DataParallel(self.net)
        #     cudnn.benchmark = True

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=1e-2, momentum=0.9, weight_decay=l2)
        self.noise_multiplier, self.max_grad_norm = noise, grad_norm
        self.net = self.net.to(self.device)
        
        if self.use_DP == 1:
            self.net = module_modification.convert_batchnorm_modules(self.net)
            inspector = DPModelInspector()
            inspector.validate(self.net)
            privacy_engine = PrivacyEngine(
                self.net,
                batch_size=64,
                sample_size=len(self.trainloader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                secure_rng=False,
            )
            print( 'noise_multiplier: %.3f | max_grad_norm: %.3f' % (self.noise_multiplier, self.max_grad_norm))
            privacy_engine.attach(self.optimizer)

        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 75], 0.1)

    # Training
    def train(self):

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if isinstance(targets, list):
                targets = targets[0]
             
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()

        if self.use_DP == 1:
            epsilon, best_alpha = self.optimizer.privacy_engine.get_privacy_spent(1e-5)
            print("\u03B1: %.3f \u03B5: %.3f \u03B4: 1e-5" % (best_alpha, epsilon))
            if self.logger is not None:
                self.logger.info("privacy budget:\u03B1: %.3f \u03B5: %.3f \u03B4: 1e-5" % (best_alpha, epsilon))
                
        self.scheduler.step()

        # print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))
        return 1.*correct/total, loss.item()/total

    def saveModel(self, path):
        torch.save(self.net.state_dict(), path)

    def get_noise_norm(self):
        return self.noise_multiplier, self.max_grad_norm

    def test(self):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.testloader:
                if isinstance(targets, list):
                    targets = targets[0]

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)

                correct += predicted.eq(targets).sum().item()

            # print( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
        return 1.*correct/total, loss.item()/total


def train_model(writer, logger, MODEL_PATH, model, train_set, test_set\
                        ,use_DP, noise, l2, \
                        batch_size, epoch, device='cuda'):
    
    attr = MODEL_PATH.split('/')[-1].split('_')[-1][:-4]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)

    model = model_training(train_loader, test_loader, model, use_DP, noise, l2, logger, device)
    acc_train = 0
    acc_test = 0

    for i in range(epoch):
        acc_train, loss_train = model.train()
        acc_test, loss_test = model.test()
        overfitting = round(acc_train - acc_test, 6)
        if i % 10 == 0:
            print('Epoch:%d %s_model, Acc_train:%.4f, Loss_train:%.4f, Acc_test:%.4f, Loss_test:%.4f, Overfit:%.4f' \
                % (i, attr, acc_train, loss_train, acc_test, loss_test, overfitting))

        logger.debug('%s_model Epoch:%d, Acc_train:%.4f, Loss_train:%.4f, Acc_test:%.4f, Loss_test:%.4f, Overfit:%.4f' \
            % (attr, i+1, acc_train, loss_train, acc_test, loss_test, overfitting))

        writer.add_scalar('Loss/%s_train' % attr, loss_train, i)
        writer.add_scalar('Loss/%s_test' % attr, loss_test, i)

        writer.add_scalar('Acc/%s_train' % attr, acc_train, i)
        writer.add_scalar('Acc/%s_test' % attr, acc_test, i)

        writer.add_scalar('%s_overfit' % attr, overfitting, i)

    model.saveModel(MODEL_PATH)
    print("Saved %s model, path:%s" % (attr, MODEL_PATH))
    return acc_train, acc_test, overfitting

def test_model(logger, target_path, target_model, target_test, device='cuda'):

    target_model = target_model.to(device)
    target_model.load_state_dict(torch.load(target_path, map_location=device))    
    target_model.eval()

    correct = 0
    total = 0
    testloader = torch.utils.data.DataLoader(target_test, batch_size=64, shuffle=False, num_workers=2)
    with torch.no_grad():
        for inputs, targets in testloader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = target_model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 1.0*correct/total
    return acc

def get_loss(target_model, target_test, filename, device):
    from utils import categorial_cross_entropy_torch
    target_model.eval()
    dataloader = torch.utils.data.DataLoader(target_test, batch_size=256, shuffle=False, num_workers=2)
    
    target_model.to(device)
    t_outputs, t_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            if isinstance(targets, list):
                targets = targets[0]
            inputs, targets = inputs.to(device), targets.to(device)
            # target_model
            outputs = target_model(inputs)
            _, predicted = outputs.max(1)
            t_outputs = outputs if t_outputs == [] else torch.cat((t_outputs, outputs), 0)
            t_targets = targets if t_targets == [] else torch.cat((t_targets, targets), 0)

    t_loss = categorial_cross_entropy_torch(t_outputs.cpu().detach().numpy(), t_targets.cpu().detach().numpy())
    
    with open(filename, 'w') as f:
      for i in t_loss:
        f.write(str(i))
        f.write('\n')
      f.close()
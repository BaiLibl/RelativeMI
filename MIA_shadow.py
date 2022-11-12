
# refer: https://github.com/liuyugeng/ML-Doctor

import os
import glob
from unittest import result
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
np.set_printoptions(threshold=np.inf)

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score, roc_curve

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class attack_for_blackbox():
    def __init__(self, attack_train_loader, attack_test_loader, target_model, shadow_model, attack_model, device='cuda'):
        self.device = device

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.0005, momentum=0.99, weight_decay=1e-5)

    def _get_data(self, model, inputs):
        result = model(inputs)
        
        output, _ = torch.sort(result, descending=True)
        _, predicts = result.max(1)
        
        prediction = []
        for predict in predicts:
            prediction.append([predict,] if predict else [0,])

        prediction = torch.Tensor(prediction)
        return output, prediction


    def train(self):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        for inputs, targets, members in self.attack_train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            output, prediction = self._get_data(self.shadow_model, inputs)
            output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

            results = self.attack_model(output, prediction)
            results = F.softmax(results, dim=1)

            losses = self.criterion(results, members)

            # compute gradient and do SGD step
            self.optimizer.zero_grad() # if not, loss constant, acc=0.5
            losses.backward()
            self.optimizer.step()

            train_loss += losses.item()
            prob, predicted = results.max(1)
            total += members.size(0)
            correct += predicted.eq(members).sum().item()
            
            final_train_gndtrth.append(members)
            final_train_predict.append(predicted)
            final_train_probabe.append(prob)

            batch_idx += 1

        final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
        final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
        final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

        acc = 1.*correct/(1.0*total)
        loss = 1.*train_loss/batch_idx

        return acc, loss

    def test(self):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0
        test_loss = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        with torch.no_grad():
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs)
                output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                results = self.attack_model(output, prediction)
                results = F.softmax(results, dim=1)

                total += members.size(0)
                prob, predicted = results.max(1)

                losses = self.criterion(results, members)
                test_loss += losses.item()
                correct += predicted.eq(members).sum().item()

                final_test_gndtrth.append(members)
                final_test_predict.append(predicted)
                final_test_probabe.append(prob)

                batch_idx += 1

        final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().detach().numpy()
        final_test_predict = torch.cat(final_test_predict, dim=0).cpu().detach().numpy()
        final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().detach().numpy()

        acc1 = accuracy_score(final_test_gndtrth, final_test_predict)
        # f1 = f1_score(final_test_gndtrth, final_test_predict)
        # auc = roc_auc_score(final_test_gndtrth, final_test_probabe)
        # recall = recall_score(final_test_gndtrth, final_test_predict)
        # precision = precision_score(final_test_gndtrth, final_test_predict)
        f1 = 0.0
        auc = 0.0
        recall = 0.0
        precision = 0.0

        fpr, tpr, _ = roc_curve(final_test_gndtrth, final_test_predict, pos_label=1)
        attack_adv = tpr[1] - fpr[1]

        acc = 1.*correct/(1.0*total)
        loss = 1.*test_loss/batch_idx

        results = [loss, acc, f1, auc, recall, precision, attack_adv]
        results = [round(i, 4) for i in results]
        return results

    def saveModel(self, path):
        torch.save(self.attack_model.state_dict(), path)
    
    def loadModel(self, path):
        self.attack_model.load_state_dict(torch.load(path, map_location=self.device))


def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)
    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)


    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))

    print(len(mem_train), len(nonmem_train), len(mem_test), len(nonmem_test))

    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader

def get_attack_dataset(target_train, target_test, batch_size):
    mem_test, nonmem_test =  list(target_train), list(target_test)

    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)

    print(len(mem_test), len(nonmem_test))

    attack_trainloader = torch.utils.data.DataLoader(
        mem_test, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        nonmem_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader

# black shadow
def black_shadow(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, attack_model_path, epoch, device='cuda'):

    attack = attack_for_blackbox(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, device)
    for i in range(epoch):
        
        acc_train, loss_train = attack.train()
        loss_test, acc_test, f1, auc, recall, precision, adv = attack.test()
        
        if i%10 == 0:
          print('Epoch %d: acc:%.4f, f1:%.4f, auc:%.4f, adv:%.4f' % (i, acc_test, f1, auc, adv))

    attack.saveModel(attack_model_path)
    print("Saved Attack Model")

def shadow_eval(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model, attack_model_path):

    attack = attack_for_blackbox(attack_trainloader, attack_testloader, target_model, shadow_model, attack_model)
    attack.loadModel(attack_model_path)

    result = attack.test()
    return result
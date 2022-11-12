
# refer: https://github.com/inspire-group/membership-inference-evaluation

from unittest import result
import numpy as np
import torch
import os

from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score, roc_curve

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes

        self.target_train_performance = target_train_performance
        self.target_test_performance = target_test_performance
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        

        mem_label = np.array([1 for i in range(len(self.t_tr_labels))])
        nonmem_label = np.zeros((len(self.t_te_labels),),dtype=int)
        gnd_label = np.append(mem_label, nonmem_label)
        pre_label = np.append(self.t_tr_corr, self.t_te_corr)

        result = self.evaluate_metric(gnd_label, pre_label)
        acc, f1, auc, recall, precision, adv = result

        print('%s: acc:%.4f, f1:%.4f, auc:%.4f, R:%.4f, P:%.4f, adv:%.4f' % ('correctness', acc, f1, auc, recall, precision, adv))
        return ['corr', result]
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        pre_label_tr = np.zeros(len(self.t_tr_labels), dtype=int)
        pre_label_te = np.zeros(len(self.t_te_labels), dtype=int)
        for num in range(self.num_classes): # set threshold by shadow model
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)

            tr = [1 if (self.t_tr_labels[i] == num and t_tr_values[i] >= thre) else 0 for i in range(len(pre_label_tr))]
            te = [1 if (self.t_te_labels[i] == num and t_te_values[i] >= thre) else 0 for i in range(len(pre_label_te))]
            pre_label_te += np.array(te)
            pre_label_tr += np.array(tr)

        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))

        mem_label = np.array([1 for i in range(len(self.t_tr_labels))])
        nonmem_label = np.zeros((len(self.t_te_labels),),dtype=int)
        gnd_label = np.append(mem_label, nonmem_label)
        pre_label = np.append(pre_label_tr, pre_label_te)

        result = self.evaluate_metric(gnd_label, pre_label)
        acc, f1, auc, recall, precision, adv = result

        print('%s: acc:%.4f, f1:%.4f, auc:%.4f, R:%.4f, P:%.4f, adv:%.4f' % (v_name, acc, f1, auc, recall, precision, adv))

        return [v_name, result]
    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        res = []
        if (all_methods) or ('correctness' in benchmark_methods):
            res.append(self._mem_inf_via_corr())
        if (all_methods) or ('confidence' in benchmark_methods):
            res.append(self._mem_inf_thre('conf', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf))
        if (all_methods) or ('entropy' in benchmark_methods):
            res.append(self._mem_inf_thre('ent', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr))
        if (all_methods) or ('modified entropy' in benchmark_methods):
            res.append(self._mem_inf_thre('ment', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr))

        return res

    def evaluate_metric(self, final_test_gndtrth, final_test_predict):

        acc = accuracy_score(final_test_gndtrth, final_test_predict)
        # f1 = f1_score(final_test_gndtrth, final_test_predict)
        # auc = roc_auc_score(final_test_gndtrth, final_test_probabe)
        f1 = 0.0
        auc = 0.0
        recall = 0.0
        precision = 0.0
        # recall = recall_score(final_test_gndtrth, final_test_predict)
        # precision = precision_score(final_test_gndtrth, final_test_predict)

        fpr, tpr, _ = roc_curve(final_test_gndtrth, final_test_predict, pos_label=1)
        attack_adv = tpr[1] - fpr[1]
        

        return round(acc,4), round(f1,4), round(auc,4), round(recall,4), round(precision,4), round(attack_adv,4)

def get_output(device, net, test_data):
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=2)
    entire_output = []
    entire_target = []
    net.to(device)
    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            if isinstance(targets, list):
                targets = targets[0]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            entire_output = outputs if entire_output == [] else torch.cat((entire_output, outputs), 0)
            entire_target = targets if entire_target == [] else torch.cat((entire_target, targets), 0)

    return [entire_output.detach().cpu().numpy(), entire_target.detach().cpu().numpy()]


def metric_eval(target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, num_classes, device):
    
    target_train_performance = get_output(device, target_model, target_train)
    target_test_performance  = get_output(device, target_model, target_test)
    shadow_train_performance = get_output(device, shadow_model, shadow_train)
    shadow_test_performance  = get_output(device, shadow_model, shadow_test)

    MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance,num_classes=num_classes)
    return MIA._mem_inf_benchmarks()
   

# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
sys.dont_write_bytecode = True
import os

import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from dnn.data_loader import Data_Loader
from dnn.model import Model
from fgsm import FGSM
from PGD import PGD
from mim import MIM

# =================================== 参数配置 =================================== #
param_list = []
# eps_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
eps_list = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]

for eps in eps_list:
    param_config = {
        'device': 'cuda:1',
        'model': 'resnet50_cifar100', # 'lenet_mnist', 'cnn_fashionmnist', 'resnet18_cifar10', 'resnet50_cifar100'
        'train_batch_size': 100,
        'test_batch_size': 100,
        'dataset': 'cifar100', # 'mnist', 'fashionmnist', 'cifar10', 'cifar100'
        'data_path': '../data/',
        # 'attack_method': ['fgsm', {'eps': eps, 'norm': np.inf}],
        # 'attack_method': ['pgd', {'eps': eps, 'step_size': 0.0078, 'num_iter': 10, 'norm': np.inf}],
        'attack_method': ['mifgsm', {'eps': eps, 'num_iter': 10, 'decay': 0.9, 'norm': np.inf}],
        'trained_model_path': '../res/exp_res/train_type-standard_model-resnet50_cifar100_train_batch_size-128_num_epochs-100_.pt',
        'res_path': './res/',
    }
    param_list.append(param_config)
# =================================== 参数配置 =================================== #
def test_attack(param_config):
    net = Model(param_config['model'], param_config['device']).load_model()
    net.load_state_dict(torch.load(param_config['trained_model_path']))
    net.eval()

    if param_config['attack_method'][0] == 'fgsm':
        adv_generater = FGSM(net=net, p=param_config['attack_method'][1]['norm'], eps=param_config['attack_method'][1]['eps'], data_name=param_config['dataset'], target=None, loss='ce', device=param_config['device'])
        param = 'eps=' + str(param_config['attack_method'][1]['eps']) + ',norm=' + str(param_config['attack_method'][1]['norm'])
    elif param_config['attack_method'][0] == 'pgd':
        adv_generater = PGD(net=net, epsilon=param_config['attack_method'][1]['eps'], norm = param_config['attack_method'][1]['norm'], stepsize=param_config['attack_method'][1]['step_size'], steps=param_config['attack_method'][1]['num_iter'], data_name=param_config['dataset'], target=None, loss='ce', device=param_config['device'])
        param = 'eps=' + str(param_config['attack_method'][1]['eps']) + ',step_size=' + str(param_config['attack_method'][1]['step_size'])  + ',num_iter=' + str(param_config['attack_method'][1]['num_iter']) + ',norm=' + str(param_config['attack_method'][1]['norm'])
    elif param_config['attack_method'][0] == 'mifgsm':
        adv_generater = MIM(net=net, epsilon=param_config['attack_method'][1]['eps'], p=param_config['attack_method'][1]['norm'], stepsize=param_config['attack_method'][1]['eps']/param_config['attack_method'][1]['num_iter'], steps=param_config['attack_method'][1]['num_iter'], decay_factor=param_config['attack_method'][1]['decay'], data_name=param_config['dataset'], target=None, loss='ce', device=param_config['device'])
        param = 'eps=' + str(param_config['attack_method'][1]['eps']) + ',num_iter=' + str(param_config['attack_method'][1]['num_iter']) + ',decay=' + str(param_config['attack_method'][1]['decay']) + ',norm=' + str(param_config['attack_method'][1]['norm'])

    train_dataloader, test_dataloader = Data_Loader(param_config['dataset'], param_config['data_path'], param_config['train_batch_size'], param_config['test_batch_size']).load_data()

    clean_train_loss = 0
    clean_test_loss = 0
    clean_train_correct = 0
    clean_test_correct = 0

    adv_train_loss = 0
    adv_test_loss = 0
    adv_train_correct = 0
    adv_test_correct = 0

    for data, target in train_dataloader:
        data, target = data.to(param_config['device']), target.to(param_config['device'])
        output_clean = net(data)
        loss = nn.CrossEntropyLoss()(output_clean, target)
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        clean_train_loss += loss.item()
        clean_train_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()

        x_adv = adv_generater.forward(data, target, target_labels=None)
        output_adv = net(x_adv)
        loss = nn.CrossEntropyLoss()(output_adv, target)
        pred_adv = output_adv.argmax(dim=1, keepdim=True)
        adv_train_loss += loss.item()
        adv_train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()
    
    clean_train_loss /= len(train_dataloader)
    clean_train_acc = clean_train_correct / len(train_dataloader.dataset)
    adv_train_loss /= len(train_dataloader)
    adv_train_acc = adv_train_correct / len(train_dataloader.dataset)

    for data, target in test_dataloader:
        data, target = data.to(param_config['device']), target.to(param_config['device'])
        output_clean = net(data)
        loss = nn.CrossEntropyLoss()(output_clean, target)
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        clean_test_loss += loss.item()
        clean_test_correct += pred_clean.eq(target.view_as(pred_clean)).sum().item()

        x_adv = adv_generater.forward(data, target, target_labels=None)
        output_adv = net(x_adv)
        loss = nn.CrossEntropyLoss()(output_adv, target)
        pred_adv = output_adv.argmax(dim=1, keepdim=True)
        adv_test_loss += loss.item()
        adv_test_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()
    
    clean_test_loss /= len(test_dataloader)
    clean_test_acc = clean_test_correct / len(test_dataloader.dataset)
    adv_test_loss /= len(test_dataloader)
    adv_test_acc = adv_test_correct / len(test_dataloader.dataset)

    if not os.path.exists(param_config['res_path'] + param_config['attack_method'][0] + '_' + param_config['model'] + '_.csv'):
        res_column = pd.DataFrame(data=[['param', 'clean_train_loss', 'adv_train_loss', 'clean_test_loss', 'adv_test_loss', 'clean_train_acc', 'adv_train_acc', 'clean_test_acc', 'adv_test_acc']])
        res_column.to_csv(param_config['res_path'] + param_config['attack_method'][0] + '_' + param_config['model'] + '_.csv', mode='w', header=False, index=False)

    res = pd.DataFrame(data=[[param, clean_train_loss, adv_train_loss, clean_test_loss, adv_test_loss, clean_train_acc, adv_train_acc, clean_test_acc, adv_test_acc]])
    res.to_csv(param_config['res_path'] + param_config['attack_method'][0] + '_' + param_config['model'] + '_.csv', mode='a', header=False, index=False)

    print('Train Loss: {}/{}\nTest Loss: {}/{}\nTrain Acc: {}/{}\nTest ACC: {}/{}'.format(clean_train_loss, adv_train_loss, clean_test_loss, adv_test_loss, clean_train_acc, adv_train_acc, clean_test_acc, adv_test_acc))

for ele in param_list:
    test_attack(ele)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Fed import net_diff, compute_2_norm_square
import numpy as np
import random
from sklearn import metrics
import copy


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):

    def __init__(self, args, dataset=None, idxs=None, fl_method='fedavg', previous_net=None):

        self.loss_func = nn.CrossEntropyLoss()
        self.args = args
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.fl_method = fl_method
        self.prox_weight_decay = args.prox_weight_decay
        self.previous_net = previous_net
        self.dataset = args.dataset
        self.model = args.model
        self.epoch_test = args.epoch_hk_test

    def test_Hk(self, net, lr):

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        hn = []

        for iter in range(self.epoch_test):

            for batch_idx, (images, labels) in enumerate(self.ldr_train):

                pre_net = copy.deepcopy(net.state_dict())

                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                hn.append(net_diff(pre_net, net.state_dict(), lr))

        return hn

    def train(self, net, lr):

        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        # optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

        # train and update
        if self.fl_method == 'fedavg':

            epoch_loss = []
            for iter in range(self.args.local_ep):

                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        elif self.fl_method == 'fedprox':

            epoch_loss = []
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.ldr_train):

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)

                    prox_term = compute_2_norm_square(net_diff(self.previous_net, net.state_dict(), 1.0))

                    '''
                    prox_term = torch.tensor(0.)
                    for param_index, param in enumerate(net.parameters()):
                        print('para: {}'.format(param))
                        print('index: {}'.format(param_index))
                        print('list: {}'.format(list(net.state_dict().keys())[param_index]))
                        print('{}'.format(self.previous_net[list(net.state_dict().keys())[param_index]]))

                        prox_term += (torch.norm((param - self.previous_net[list(net.state_dict().keys())[param_index]])) ** 2)
                    '''

                    loss = self.loss_func(log_probs, labels) + self.prox_weight_decay / 2 * prox_term
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        else:
            epoch_loss = []
            exit('unrecognized fl method')

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def test_loss(self, net):

        net.eval()

        test_loss = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            probs = net(images)
            # sum up the batch loss
            test_loss += self.loss_func(probs, labels).item()

        test_loss /= len(self.ldr_train)

        return test_loss

    def test_output(self, net, test_dataset, one_hot_encoded_ind=True):

        net.eval()

        label_list = list()

        for batch_idx, (images, labels) in enumerate(test_dataset):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            log_probs = net(images)
            if one_hot_encoded_ind:
                y_pred = log_probs.data.max(1, keepdim=True)[1]
                label_list.extend(y_pred)
            else:
                label_list.extend(log_probs)

        return label_list



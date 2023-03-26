#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np


def FedAvg(nn_weight, w_locals):

    nn_weight = nn_weight / sum(nn_weight)

    w_avg = copy.deepcopy(w_locals[0])     # obtain the first weight
    for k in w_avg.keys():

        try:
            w_avg[k] *= nn_weight[0]
        except RuntimeError:
            w_avg[k] = float(w_avg[k])
            w_avg[k] *= nn_weight[0]
            w_avg[k] = torch.tensor(w_avg[k])

        for i in range(1, len(w_locals)):
            w_avg[k] += w_locals[i][k] * nn_weight[i]

    return w_avg


def net_diff(pre_net, after_net, lr):
    diff = copy.deepcopy(pre_net)
    for k in pre_net.keys():
        diff[k] -= after_net[k]
        if str(diff[k].dtype) == str(torch.tensor(50).dtype):
            diff[k] = float(diff[k])
        diff[k] /= lr
    return diff


def compute_2_norm_square(model):
    norm = 0
    for kk in model.keys():
        try:
            norm += torch.norm(model[kk], p=2) ** 2
        except AttributeError:
            norm += model[kk] ** 2

    return norm


def net_norm(hk_locals):

    user_num = len(hk_locals)

    hk_avg = FedAvg(np.ones(user_num), hk_locals)
    hk = 0
    for uu in range(user_num):
        hk += compute_2_norm_square(net_diff(hk_locals[uu], hk_avg, 1))

    hk /= (user_num - 1)

    return hk


def gradient_norm(hk_locals):
    epoch_hk_test = len(hk_locals[0])
    tmp_HK = np.zeros(epoch_hk_test)
    for ee in range(epoch_hk_test):
        tmp_hk_locals = list()
        for uu in range(len(hk_locals)):
            tmp_hk_locals.append(hk_locals[uu][ee])

        tmp_HK[ee] = net_norm(tmp_hk_locals)

    return np.max(tmp_HK)

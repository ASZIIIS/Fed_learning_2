#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
import torch

import sys
sys.path.append("..")


from utils.options import args_parser
from utils.settings import set_args, set_model, set_dataset, set_lr, set_epoch, get_user_availability
from utils.sampling import user_sampling_round
from models.Update import LocalUpdate
from models.Fed import FedAvg, gradient_norm
from models.test import test_img
import random

random.seed(10)
np.random.seed(10)


def client_weight(dict_user):
    num = list()
    for ii in range(len(dict_user)):
        num.append(len(dict_user[ii]))

    num = np.array(num)
    num = num / sum(num)
    num = np.array(num)

    return num


def strata_weight(nn_weight, strata):
    ss_weight = np.zeros(len(strata))
    for ii in range(len(strata)):
        ss_weight[ii] = np.sum(nn_weight[np.array(strata[ii])])

    return ss_weight / sum(ss_weight)


def run(args, file_index=0, fl_method='fedavg', strat_ind=False, round_ind=None, allocation=None):

    # file_name
    file_name = './save/{}_{}_{}_stra{}_{}_n{}f{}e{}b{}g{}noniid{}lr{}_{}.txt'.format(
        args.dataset, fl_method, int(args.prox_weight_decay * 10000), str(strat_ind), str(allocation),
        int(args.num_users), int(args.frac * 100), args.local_ep, args.local_bs, args.avil_trace_group,
        int(args.noniid_frac * 100), int(args.lr * 1000000), file_index
    )

    # build model
    print('Initializing model ...')
    net_glob = set_model(args)

    # initialization
    print('Training ... ')
    net_glob.train()
    nn_weight = client_weight(dict_users)
    ss_weight = strata_weight(nn_weight, strata)

    # compute Hk ==========================================================================================
    if strat_ind == True and allocation == 'optimal':
        print('Test Hk')
        hk_locals = dict()
        HK = np.zeros(len(strata))
        for kk in range(len(strata)):
            print('strata idex: {}'.format(kk))
            hk_locals[kk] = list()
            for ii in range(len(strata[kk])):
                idxs = random.choices(dict_users[strata[kk][ii]], k=args.local_bs)
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=idxs)
                hk = local.test_Hk(net=copy.deepcopy(net_glob).to(args.device), lr=args.lr)
                hk_locals[kk].append(hk)
            HK[kk] = gradient_norm(hk_locals[kk])

        HK /= sum(HK)
        print('HK: {}'.format(HK))
    else:
        HK = np.zeros(len(strata))

    # training ===============================================================================================
    global_loss = []
    learning_rate = args.lr

    for iter in range(args.epochs):  # for each epoch

        print(iter)

        # client sampling ============================================================================
        sample_users = user_sampling_round(args, user_availability[:, iter], strat_ind=strat_ind, strata=strata,
                                           round_ind=round_ind, allocation=allocation, Hk=HK)
        w_locals, loss_locals = [], []

        # download, local train =====================================================================
        for idx in sample_users:  # initialize a class LocalUpdate

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[int(idx)],
                                fl_method=fl_method, previous_net=net_glob.state_dict())
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), lr=learning_rate)
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        global_loss.append(np.mean(np.array(loss_locals)))

        # update global model =======================================================================
        if not strat_ind:
            w_glob = FedAvg(nn_weight[sample_users], w_locals)
        else:
            w_strata = list()
            for ii in range(len(strata)):
                strata_sample_users = set(np.array(strata[ii])) & set(np.array(sample_users))
                tmp_user = list()
                for uu in strata_sample_users:
                    tmp_user.append(np.where(sample_users == uu)[-1][0])

                if len(tmp_user) > 0:
                    tmp_w_locals = list()
                    for uu in tmp_user:
                        tmp_w_locals.append(w_locals[uu])
                    tmp_w = FedAvg(nn_weight[sample_users[tmp_user]], tmp_w_locals)
                    w_strata.append(tmp_w)
                else:
                    w_strata.append(net_glob.state_dict())

            w_glob = FedAvg(ss_weight, w_strata)

        # for kk in net_glob.state_dict().keys():
            # print('net_glob: {}'.format(net_glob.state_dict()[kk]))
            # print('after: {}'.format(w_glob[kk]))

        net_glob.load_state_dict(w_glob)

        # obtain and print loss ============================================================================
        if iter % args.store_iter == 0:
            print(iter)
            net_glob.eval()
            accuracy, loss = test_img(net_glob, dataset_train, args)
            print('Round {:3d}, Average loss {:.6f}, Accuracy {:.3f}'.format(iter, loss, accuracy))
            with open(file_name, 'a') as file_object:
                file_object.write('{:.6f}\t{:.6f}\t'.format(loss, accuracy))

            accuracy, loss = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Test Average loss {:.6f}, Test Accuracy {:.3f}'.format(iter, loss, accuracy))
            with open(file_name, 'a') as file_object:
                file_object.write('{:.6f}\t{:.6f}\n'.format(loss, accuracy))


if __name__ == '__main__':

    # parse args
    args = args_parser()
    args.gpu = -1
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.store_iter = 1
    args.prox_weight_decay = 0.01  # for FedProx
    args.max_strata = 20

    # model and dataset
    args.dataset = 'cifar' # 'cifar'  # {'mnist', 'cifar'}
    args.local_ep = 5  # number of local epoch

    args.num_users = 100
    args.frac = 0.1

    args.noniid_frac = 0.5  # fraction of non-iid data
    args.avil_trace_group = 1
    args.lr = set_lr(args)
    method = 'fedprox'

    args.epochs = 200 # set_epoch(args)
    # args.epochs = 20
    args.epoch_hk_test = 5
    args = set_args(args)

    # run simulation
    # set dataset
    print('Initializing dataset ...')
    dataset_train, dataset_test, dict_users, strata = set_dataset(args)
    user_availability = get_user_availability(strata, args.num_users, args.epochs, args.avil_trace_group)

    for file_index in range(5):

       # run(args, file_index=file_index, fl_method=method, strat_ind=True, round_ind='random',
       #     allocation='optimal')

        run(args, file_index=file_index, fl_method=method, strat_ind=True, round_ind='random',
            allocation='proportional')

        run(args, file_index=file_index, fl_method=method, strat_ind=False)

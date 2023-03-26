#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from models.Update import LocalUpdate
from sklearn import metrics
import copy
from models.Nets import CNNMnist, CNNCifar
from torchvision import datasets, transforms, models
import torch
import math
import torch.nn.init as init
np.set_printoptions(threshold=np.inf)

"""
SET PARAMETERS =======================
"""

def set_args(args):

    if args.dataset == 'mnist':
        args.num_channels = 1
        args.local_bs = 10
    elif args.dataset == 'cifar':
        args.local_bs = 50
        args.model = 'resnet'
    else:
        net_glob = {}
        exit('Error: unrecognized model')

    return args


def set_lr(args):

    lr = args.lr

    if args.dataset in ['mnist']:
        if args.noniid_frac == 1:
            lr = 0.05
        elif args.noniid_frac == 0.95:
            lr = 0.03
        elif args.noniid_frac <= 0.8:
            lr = 0.01
        else:
            print('undefined lr')

    elif args.dataset in ['cifar']:
        if args.noniid_frac == 1:
            lr = 0.065
        elif args.noniid_frac == 0.95:
            lr = 0.065
        elif args.noniid_frac <= 0.8:
            lr = 0.05
        else:
            print('undefined lr')

    return lr


def set_epoch(args):

    epoch = args.epochs
    if args.dataset in ['mnist', 'cifar']:
        if args.noniid_frac == 1:
            epoch = 3000
        elif args.noniid_frac == 0.95:
            epoch = 400
        elif args.noniid_frac == 0.8:
            epoch = 400
        elif args.noniid_frac == 0:
            epoch = 200
        else:
            print('undefined epoch')

    return epoch


def set_dataset(args):

    # collect dataset
    if args.dataset == 'mnist':

        # load dataset 'mnist' and split users ==============================================================
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)

        # dict_users, strata
        dict_users = mnist_user_dataset(dataset_train, args.num_users, args.noniid_frac)
        strata = mnist_user_strata(dataset_train, dict_users, args)

    elif args.dataset == 'cifar':

        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./data/cifar/', train=False, download=True, transform=trans_cifar)

        # dict_users, strata
        dict_users = cifar_user_dataset(dataset_train, args.num_users, args.noniid_frac)
        strata = cifar_user_strata(dataset_train, dict_users, args)

    else:
        dataset_train = {}
        dataset_test = {}
        dict_users = {}
        strata = {}
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users, strata


def set_model(args):

    if args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'cifar' and args.model == 'cnn':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.dataset == 'cifar' and args.model == 'resnet':
        # net_glob = torch.hub.load('pytorch/vision:v0.8.0', 'resnet18', pretrained=True)
        net_glob = models.resnet34(pretrained=False)
        net_glob.load_state_dict(torch.load('./models/resnet34-333f7ec4.pth'))
    else:
        net_glob = {}
        exit('Error: unrecognized model')

    print(net_glob)

    return net_glob


"""
USER STRATA  ==============================
"""

def user_strata(user_prob_matrix, args):
    """
    :param user_prob_matrix:
    :return: strata
    """

    strata_list = list()
    score_list = list()

    user_prob_matrix = np.array(user_prob_matrix)

    # avoid duplicate elements, add some very small values
    row, column = np.shape(user_prob_matrix)
    user_prob_matrix += np.random.rand(row, column) * 1e-5

    for num_strata in range(2, args.max_strata):

        # GaussianMixture [sample, feature]
        clf = GaussianMixture(n_components=num_strata)
        clf.fit(user_prob_matrix)
        user_strata_index = clf.predict(user_prob_matrix)

        # strata: user_set = strata[strata_index]
        tmp_strata = list()
        for ii in range(num_strata):
            tmp_set = np.where(user_strata_index == ii)
            tmp_strata += tmp_set

        strata_list.append(tmp_strata) # each strata could have no client
        score_list.append(metrics.silhouette_score(user_prob_matrix, user_strata_index))

    tmp_max = int(np.argmax(np.array(score_list)))
    if score_list[tmp_max] >= 0.5:
        strata = strata_list[tmp_max]
        print('Number of Strata: {}'.format(tmp_max + 2))
    else:
        strata = list()
        strata.append(np.array(range(0, args.num_users)))
        print('Number of Strata: {}'.format(1))

    print('score_list: {}'.format(np.array(score_list)))
    # for ii in range(len(strata_list)):
    #     print('strata_list: {}'.format(strata_list[ii]))

    '''# check strata
    print('strata')
    for test_ii in range(len(strata)):
        print(len(strata[test_ii]))
        print(strata[test_ii])
        for ii in range(len(strata[test_ii])):
            print('strata: {}, {:.2f}'.format(strata[test_ii][ii], user_prob_matrix[strata[test_ii][ii], :]))
    '''
    return strata


def mnist_user_dataset(dataset, num_users, noniid_fraction):
    """
    Sample a 'fraction' of non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param fraction:
    :return:
    """

    # initialization
    total_items = len(dataset)
    num_noniid_items = len(dataset) * noniid_fraction
    num_iid_items = total_items - num_noniid_items
    dict_users = list()
    for ii in range(num_users):
        dict_users.append(list())
    idxs = [i for i in range(len(dataset))]

    # IID
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users)
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += tmp_set
            idxs = list(set(idxs) - tmp_set)

    # NON-IID
    if num_noniid_items != 0:

        num_shards = num_users  # each user has one shard
        per_shards_num_imgs = int(num_noniid_items / num_shards)
        idx_shard = [i for i in range(num_shards)]
        labels = dataset.train_labels[idxs].numpy()

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        i = 0
        while idx_shard:
            rand_idx = np.random.choice(idx_shard, 1, replace=False)
            idx_shard = list(set(idx_shard) - set(rand_idx))
            dict_users[i].extend(idxs[int(rand_idx) * per_shards_num_imgs: (int(rand_idx) + 1) * per_shards_num_imgs])
            i = divmod(i + 1, num_users)[1]

    return dict_users


def mnist_user_strata(dataset, dict_users, args, dataset_label=False):
    """
        Partition clients into strata based on their dataset
        :param dataset:
        :param dict_users:
        :param num_user:
        :param num_strata:
        :return:
    """

    # obtain label set
    label_set = [i for i in range(10)]
    num_users = len(dict_users)
    if not dataset_label:
        dataset_label = dataset.train_labels
    else:
        dataset_label = dataset

    # obtain probability of each label
    user_prob_matrix = np.zeros((num_users, len(label_set)))
    for nn in range(num_users):
        for ii in range(len(dict_users[nn])):

            tmp_label = int(dataset_label[dict_users[nn][ii]].numpy())
            user_prob_matrix[nn, tmp_label] = user_prob_matrix[nn, tmp_label] + 1

        user_prob_matrix[nn, :] = user_prob_matrix[nn, :] / sum(user_prob_matrix[nn, :])

    strata = user_strata(user_prob_matrix, args)

    return strata


def cifar_user_dataset(dataset, num_users, noniid_fraction):

    """
    Sample a 'fraction' of non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :param fraction:
    :return:
    """

    # initialization
    total_items = len(dataset)
    num_noniid_items = len(dataset) * noniid_fraction
    num_iid_items = total_items - num_noniid_items
    dict_users = list()
    for ii in range(num_users):
        dict_users.append(list())
    idxs = [i for i in range(len(dataset))]

    # IID
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users)
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += tmp_set
            idxs = list(set(idxs) - tmp_set)

    # NON-IID
    if num_noniid_items != 0:

        num_shards = num_users  # each user has one shard
        per_shards_num_imgs = int(num_noniid_items / num_shards)
        idx_shard = [i for i in range(num_shards)]
        labels = list()
        for ii in range(len(idxs)):
            labels.append(dataset[idxs[ii]][1])

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1,:].argsort()]
        idxs = idxs_labels[0, :]

        # divide and assign
        i = 0
        while idx_shard:
            rand_idx = np.random.choice(idx_shard, 1, replace=False)
            idx_shard = list(set(idx_shard) - set(rand_idx))
            dict_users[i].extend(idxs[int(rand_idx) * per_shards_num_imgs: (int(rand_idx) + 1) * per_shards_num_imgs])
            i = divmod(i + 1, num_users)[1]

    '''
    for ii in range(num_users):
        tmp = list()
        for jj in range(len(dict_users[ii])):
            tmp.append(dataset[dict_users[ii][jj]][1])
        tmp.sort()
        print(tmp)
    '''

    return dict_users


def cifar_user_strata(dataset, dict_users, args, dataset_label=False):
    """
        Partition clients into strata based on their dataset
        :param dataset:
        :param dict_users:
        :param num_user:
        :param num_strata:
        :return:
    """

    label_set = [i for i in range(10)]
    num_users = len(dict_users)
    if not dataset_label:
        dataset_label = [dataset[ii][1] for ii in range(len(dataset))]
    else:
        dataset_label = dataset

    # obtain probability of each label
    user_prob_matrix = np.zeros((num_users, len(label_set)))
    for nn in range(num_users):
        for ii in range(len(dict_users[nn])):

            tmp_label = int(dataset_label[dict_users[nn][ii]])
            user_prob_matrix[nn, tmp_label] = user_prob_matrix[nn, tmp_label] + 1

        user_prob_matrix[nn, :] = user_prob_matrix[nn, :] / sum(user_prob_matrix[nn, :])

    strata = user_strata(user_prob_matrix, args)

    return strata

'''
def real_time_user_strata(args, dataset_train, dataset_test, dict_users):

    test_epoch = 10

    # For each client, train several training round
    user_dict_list = list()
    label_set = list()
    for nn in range(args.num_users):
        tmp_net = set_model(args)
        tmp_net.train()
        for ii in range(test_epoch):
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[nn],
                                fl_method=args.fl_method, previous_net=tmp_net.state_dict())
            w, loss = local.train(net=copy.deepcopy(tmp_net).to(args.device), lr=args.lr)
            tmp_net.load_state_dict(copy.deepcopy(w))

        tmp_label_set = tmp_net.test_output(copy.deepcopy(tmp_net.state_dict()), dataset_test)
        label_set.extend(tmp_label_set)
        user_dict_list.append(range(len(label_set) - len(tmp_label_set), len(label_set)))

    # obtain strata
    if args.dataset == 'mnist':
        strata = mnist_user_strata(label_set, user_dict_list, args, dataset_label=True)
    elif args.dataset == 'cifar':
        strata = cifar_user_strata(label_set, user_dict_list, args, dataset_label=True)
    else:
        strata = {}
        exit('Error: unrecognized dataset')

    return strata
'''

"""
USER AVAILABILITY ==============================
"""


def get_user_availability(strata, num_users, num_epochs, trace_group=1):

    file_name = 'data/twitch-data/user_array_20_100_9126.txt'
    num_trace = 9126
    num_trace_per_group = int(num_trace / trace_group)

    trace_dict = np.zeros([num_users, num_epochs])
    sample_trace = np.zeros(num_users)
    # ======= select trace id ===========================
    for ii in range(len(strata)):
        tmp_group = ii % trace_group  # int(np.random.choice(range(trace_group), 1))
        tmp_sample_trace = np.random.choice(
            range(tmp_group * num_trace_per_group, (tmp_group + 1) * num_trace_per_group), len(strata[ii]),
            replace=False)

        sample_trace[np.array(strata[ii])] = tmp_sample_trace

    # ======== obtain trace, following user index =================
    f = open(file_name)
    line = f.readline()
    line_count = 0
    while line:

        user_idx = np.where(sample_trace == line_count)[-1]

        for uu in user_idx:

            num = np.array([int(x) for x in line.split('\t')])
            num_extend = [x for x in num for i in range(3)]     # each element repeats three times
            recent_num_epochs = 0
            while num_epochs - recent_num_epochs > len(num_extend):
                trace_dict[uu, recent_num_epochs: recent_num_epochs + len(num_extend)] = num_extend
                recent_num_epochs += len(num_extend)

            trace_dict[uu, recent_num_epochs:] = num_extend[:num_epochs - recent_num_epochs]

        line = f.readline()
        line_count += 1

    # shift to a random position
    for ii in range(num_users):
        trace_dict[ii, :] = np.roll(trace_dict[ii, :], np.random.randint(0, num_epochs))

    return trace_dict

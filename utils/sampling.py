import numpy as np
import random

def fractional_user_num(fractional_part):

    group_num = np.zeros(len(fractional_part))
    multipl_coeff = 10

    while True:
        if sum(np.floor(fractional_part * multipl_coeff) - fractional_part * multipl_coeff) <= 10e-5:
            break
        else:
            multipl_coeff = multipl_coeff * 10

    total_N = int(sum(fractional_part) * multipl_coeff)

    random_index = random.randint(0, total_N - 1)
    index_list = list(range(random_index, random_index + total_N, multipl_coeff))

    for ii in range(len(index_list)):

        find_group = np.where(np.cumsum(fractional_part) * multipl_coeff <= index_list[ii] % total_N)[0]

        if len(find_group) > 0:
            group_num[find_group[-1] + 1] = group_num[find_group[-1] + 1] + 1
        else:
            group_num[0] = group_num[0] + 1

    return group_num


def user_sampling_round(args, user_avb, strat_ind, strata, round_ind=None, allocation=None, Hk=None):

    def sample_round(Sk, tmp_gap):
        if round_ind == 'fixed':
            while sum(Sk) < sample_user_num:
                Sk[np.where(tmp_gap == np.max(tmp_gap))[0][0]] += 1
                tmp_gap[np.where(tmp_gap == np.max(tmp_gap))[0][0]] = 0
        elif round_ind == 'random':
            Sk = Sk + fractional_user_num(tmp_gap)
        else:
            exit('unrecognized rounding indicator')
        return Sk

    def strata_sample_num():

        Nk = np.zeros(len(strata))
        for ii in range(len(strata)):
            Nk[ii] = len(strata[ii])

        S = int(sample_user_num)

        if allocation == 'proportional':
            Sk = np.floor(args.frac * Nk)
            Sk_gap = args.frac * Nk - Sk
        elif allocation == 'optimal':
            Sk = np.floor(Hk * Nk * S / sum(Hk * Nk))
            Sk_gap = Hk * Nk * S / sum(Hk * Nk) - Sk
        else:
            Sk = 0
            Sk_gap = 0
            print('unidentified allocation scheme')

        if sum(Sk_gap) != 0:
            Sk = sample_round(Sk, Sk_gap)

        for ii in range(len(strata)):
            Sk[ii] = min(int(Sk[ii]), len(avb_user_strata[ii]))

        return Sk

    # compute S
    sample_user_num = args.frac * args.num_users

    # compute the set of available user
    available_user_set = np.unique(np.array(range(1, args.num_users + 1)) * user_avb)
    available_user_set = available_user_set[available_user_set != 0]
    available_user_set -= 1

    # compute the set of available user in each strata
    avb_user_strata = list()
    for ii in range(len(strata)):
        tmp_strata_user = np.array(strata[ii])
        avb_user_strata.append(tmp_strata_user[np.where(user_avb[tmp_strata_user] == 1)])

    # sample users
    if not strat_ind:
        sample_users = random.choices(available_user_set, k=int(sample_user_num))
    else:
        strata_sample_num = strata_sample_num()
        sample_users = list()
        for ii in range(len(strata)):
            tmp_idxs_users = np.random.choice(avb_user_strata[ii], int(strata_sample_num[ii]), replace=False)  # randomly sample a set of users
            sample_users.extend(tmp_idxs_users)

        sample_users = np.array(sample_users)

    for uu in range(len(sample_users)):
        sample_users[uu] = int(sample_users[uu])

    print('strata num: {}'.format(len(strata)))
    print('available user: {}'.format(avb_user_strata))
    print('sample user: {}'.format(sample_users))

    return sample_users

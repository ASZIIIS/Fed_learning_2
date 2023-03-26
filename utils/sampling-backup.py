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




def user_sampling_round(args, user_avb, strat_ind, strata, round_ind=None, allocation=None):

    user_num_per_stratum = np.zeros(len(strata))
    for ii in range(len(strata)):
        user_num_per_stratum[ii] = len(strata[ii])

    sample_user_num = args.frac * args.num_users
    available_user_set = np.unique(np.array(range(1, args.num_users + 1)) * user_avb)
    available_user_set = available_user_set[available_user_set != 0]
    available_user_set -= 1

    if not strat_ind:

        sample_users = random.choices(available_user_set, k=int(sample_user_num))

    else:

        strata_user_sample_num = np.floor(args.frac * user_num_per_stratum)
        tmp_gap = args.frac * user_num_per_stratum - strata_user_sample_num
        if sum(tmp_gap) != 0:
            if round_ind == 'fixed':
                while sum(strata_user_sample_num) < sample_user_num:
                    strata_user_sample_num[np.where(tmp_gap == np.max(tmp_gap))[0][0]] += 1
                    tmp_gap[np.where(tmp_gap == np.max(tmp_gap))[0][0]] = 0
            elif round_ind == 'random':
                strata_user_sample_num = strata_user_sample_num + fractional_user_num(tmp_gap)
            else:
                exit('unrecognized rounding indicator')

        sample_users = list()
        for ii in range(len(strata)):
            tmp_strata_user = np.array(strata[ii])
            tmp_avb_user_strata = tmp_strata_user[np.where(user_avb[tmp_strata_user] == 1)]
            tmp_idxs_users = np.random.choice(tmp_avb_user_strata, min(int(strata_user_sample_num[ii]),
                                                                       len(tmp_avb_user_strata)),
                                              replace=False)  # randomly sample a set of users

            sample_users.extend(tmp_idxs_users)

        sample_users = np.array(sample_users)

    return sample_users

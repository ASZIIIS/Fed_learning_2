import os
import datetime
import numpy as np
import random

random.seed(10)
np.set_printoptions(threshold=np.inf)
""" 
2017-10-05-17-30  -- 2017-10-21-16-00
every 15 minutes
[user][time_index (0-1513)] = {0,1}
"""

dt_ref = datetime.datetime(2017, 10, 5, 17, 30)
# dt_endref = datetime.datetime(2017, 10, 6, 18, 00)
dt_endref = datetime.datetime(2017, 10, 21, 16, 00)
max_time_id = int(int((dt_endref - dt_ref).total_seconds() / 60.0) / 15 + 1)

print('max_time_id: {}'.format(max_time_id))

path = './data_filtered_utc'
files = os.listdir(path)

user_dict = dict()
count = 0

for file_name in files:

    if file_name == '.DS_Store':
        continue

    f = open(path + "/" + file_name)
    print(file_name)
    line = f.readline()

    split_file_name = file_name.split('.')[0].split('-')
    dt_tmp = datetime.datetime(int(split_file_name[0]), int(split_file_name[1]), int(split_file_name[2]),
                                int(split_file_name[3]), int(split_file_name[4]))

    while line:

        split_line = line.split('\t')
        if split_line[0] != '\n':
            tmp_user_id = int(split_line[0])

            if tmp_user_id not in user_dict.keys():

                user_dict[tmp_user_id] = np.zeros(max_time_id)

            time_id = int((dt_tmp - dt_ref).total_seconds() / 60.0 / 15)
            user_dict[tmp_user_id][time_id] = int(1)

        line = f.readline()

    f.close()


###COUNT HAS ISSUE ....  CORRECT ....

'''
print(len(user_dict))
count = 0
for tmp_key in user_dict.keys():

    print(user_dict[tmp_key])
    count = count + 1

    if count > 30:
        break
'''

user_array = np.zeros([len(user_dict), max_time_id])

print(len(user_dict))

count = 0
for tmp_key in user_dict.keys():
    user_array[count, :] = user_dict[tmp_key]
    count += 1

user_avb_freq = np.sum(user_array, axis=1)
user_array = user_array[np.argsort(user_avb_freq), :]
print(np.sort(user_avb_freq[::100]))

f_name = 'user_array.txt'
np.savetxt(f_name, user_array[0::20, :], fmt='%i', delimiter='\t')

'''
with open(f_name, 'a') as file_object:
    for ii in range(len(user_dict)):
        if divmod(ii, 100) == 0:
            file_object.write(str(user_array[ii, :]))
            file_object.write('\n')
'''
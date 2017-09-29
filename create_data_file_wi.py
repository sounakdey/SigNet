import numpy as np
import random
import getpass as gp
import os
import glob

usr = gp.getuser()




dataset = 'GPDS300'
ext = '*.jpg'

# image_dir = '/home/' + usr + '/Workspace/Datasets/' + dataset + '/'
image_dir = '/home/sounak/Documents/Datasets/GPDS300/'
data_file = '/home/sounak/Documents/Datasets/GPDS300/list_genuine.txt'
data_file_forg = '/home/sounak/Documents/Datasets/GPDS300/list_forgery.txt'

subdir_names = sorted(next(os.walk(image_dir))[1])

with open(data_file) as f:
    lines = f.read().splitlines()

with open(data_file_forg) as f:
    lines_forged = f.read().splitlines()

with open('./idx_test_writers.txt') as f:                   # for train: idx_train_writers.txt
    idx_train_writers = f.read().splitlines()

target = open('gpds_pairs_icdar_test.txt', 'w')             # for test: gpds_pairs_icdar_train.txt
# target_test = open('gpds_pairs_icdar_test.txt', 'w')

for num, subdir in enumerate(subdir_names):
    except_dir = [x for x in subdir_names if x != subdir ]
    rand_writer = random.sample(except_dir, 24)
    list_diff = [line for line in lines if rand_writer.count(line.split('/')[0])]
    rand_diff = random.sample(list_diff, 276)
    list_equal = [line for line in lines if line.split('/')[0] == str(subdir)]
    list_forged = [line for line in lines_forged if line.split('/')[0] == str(subdir)]
    if idx_train_writers.count(str(int(subdir)-1)):
        for indx, i in enumerate(list_equal):
            for j in list_equal[indx+1:]:
                if i != j:
                    target.write(i+' '+j+' '+'1'+'\n')
                    # target_test.write(i+' '+j+' '+'1'+'\n')

        for diff in rand_diff:
            rand_same = random.sample(list_equal, 1)
            target.write(rand_same[0]+' '+diff+' '+'0'+'\n')

    # for indx, i in enumerate(list_equal):
    #     for j in list_forged[indx+1:]:
    #         if i != j:
    #             target_test.write(i+' '+j+' '+'0'+'\n')


target.close()
# target_test.close()



# Assuming f2 has only 1 value in each row.
# Different values are comma separated.
# Assuming f1 has osdata at 3rd place value.
# Each file has 'n' runs.
# Writing to f3 after replace f1's 3rd value with f2's 2nd.

import pandas as pd
import csv
import os

file_1_path = input('Enter absolute path to file 1: ')
file_2_path = input('Enter absolute path to file 2: ')
file_3_path = input('Enter absolute path to file 3: ')
runs = []

for fname in os.listdir(file_1_path):
    curr_run_path_1 = os.path.join(file_1_path, fname)
    curr_run_path_2 = os.path.join(file_2_path, fname)
    curr_run_path_3 = os.path.join(file_3_path, fname)

    directory = os.path.dirname(curr_run_path_3)
    if not os.path.exists(directory):
        os.makedirs(directory)

    r1 = csv.reader(open(curr_run_path_1))
    r2 = csv.reader(open(curr_run_path_2))
    w3 = csv.writer(open(curr_run_path_3, 'w+'))

    lines1 = list(r1)
    lines2 = list(r2)
    lines3 = []
    for each1, each2 in zip(lines1[1:], lines2[1:]):
        each1[2] = each2[0]

    w3.writerows(lines1)




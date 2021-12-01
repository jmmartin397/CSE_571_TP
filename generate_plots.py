import os

import matplotlib.pyplot as plt
import pandas as pd
from evaluate import RUN_DATA_DIRECTORY


def read_run_data(batch_name):
    """ return a list of DataFrames where each DataFrame contains one run's worth
    of training episode scores.
    """
    runs = []
    for fname in os.listdir(os.path.join(RUN_DATA_DIRECTORY, batch_name)):
        # fname = 'run{}.csv'.format(run_no)
        path = os.path.join(RUN_DATA_DIRECTORY, batch_name, fname)
        run = pd.read_csv(path)
        runs.append(run)
    return runs

def average_over_runs(runs):
    return sum(runs) / len(runs)


count_plots = input('Enter number of subplots: ')

row_size = input('Enter row size(default = 1): ')
col_size = input('Enter col size(default = 1): ')

if row_size == '':
    row_size = 1
if col_size == '':
    col_size = 1

row_size = int(row_size)
col_size = int(col_size)

figure, axis = plt.subplots(row_size, col_size, squeeze=False, figsize=(13,5))

for i in range(row_size):
    for j in range(col_size):
        batch_patch = input('Enter data for plot({}, {}): '.format(i, j))
        runs = read_run_data(batch_patch)
        averaged_run_data = average_over_runs(runs)

        params = {
            'alpha': .1,
            'epsilon': .05,
            'gamma': .9,
            'os_lambda': .9
        }
        input_alpha = input('Enter alpha(default = .1): ')
        input_epsilon = input('Enter epsilon(default = .05): ')
        input_gamma = input('Enter gamma(default = .9): ')
        input_os_lambda = input('Enter os_lambda(default = .9): ')

        if input_alpha != '':
            params['alpha'] = input_alpha
        if input_epsilon != '':
            params['epsilon'] = input_epsilon
        if input_gamma != '':
            params['gamma'] = input_gamma
        if input_os_lambda != '':
            params['os_lambda'] = input_os_lambda

        font_size_title = 7
        font_size_labels = 10

        title_str = ' | '.join(['{}={}'.format(key, value) for key, value in params.items()])
        axis[i, j].set_title(title_str, fontweight="bold", size=font_size_title)
        avg_over_runs = input('Enter avg over runs for this subplot(default = 100): ')
        if avg_over_runs == '':
            avg_over_runs = 100
        axis[i, j].set_ylabel('Total reward on episode (avg over {} runs)'.format(avg_over_runs), fontsize=font_size_labels)
        axis[i, j].set_xlabel('Episodes', fontsize=font_size_labels)
        # axis[i, j].legend()

        axis[i, j].spines['bottom'].set_color('#000000')
        axis[i, j].spines['top'].set_color('none')
        axis[i, j].spines['right'].set_color('none')
        axis[i, j].spines['left'].set_color('#000000')
        axis[i, j].plot(averaged_run_data)
        axis[i, j].legend()


plt.show()
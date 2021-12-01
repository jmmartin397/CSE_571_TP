import os

import matplotlib
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


matplotlib.rc('xtick', labelsize=5)
matplotlib.rc('ytick', labelsize=5)

count_plots = int(input('Enter number of subplots: '))

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
        if count_plots == 0:
            break
        count_plots -= 1
        batch_patch = input('Enter data for plot({}, {}): '.format(i, j))
        runs = read_run_data(batch_patch)
        averaged_run_data = average_over_runs(runs)

        params = {
            'Alpha': .1,
            'Epsilon': .05,
            'Gamma': .9,
            'Lambda': .9
        }
        input_alpha = input('Enter alpha(default = .1): ')
        input_epsilon = input('Enter epsilon(default = .05): ')
        input_gamma = input('Enter gamma(default = .9): ')
        input_os_lambda = input('Enter os_lambda(default = .9): ')

        if input_alpha != '':
            params['Alpha'] = input_alpha
        if input_epsilon != '':
            params['Epsilon'] = input_epsilon
        if input_gamma != '':
            params['Gamma'] = input_gamma
        if input_os_lambda != '':
            params['Lambda'] = input_os_lambda

        # avg_over_runs = input('Enter avg over runs for this subplot(default = 100): ')
        # if avg_over_runs == '':
        #     avg_over_runs = 100

        font_size_title = 9
        font_size_labels = 6
        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 7,
                }
        title_str = '  '.join(['{}={}'.format(key, value) for key, value in params.items()])
        axis[i, j].set_title(title_str, fontweight="bold", fontdict=font, bbox=dict(facecolor='none', edgecolor='grey', boxstyle='round,pad=0.5'))

        avg_over_runs = 100
        axis[i, j].set_ylabel('Total rewards on episode (avg over {} runs)'.format(avg_over_runs), fontsize=font_size_labels)
        axis[i, j].set_xlabel('Episodes', fontsize=font_size_labels)
        axis[i, j].spines['bottom'].set_color('#000000')
        axis[i, j].spines['top'].set_color('none')
        axis[i, j].spines['right'].set_color('none')
        axis[i, j].spines['left'].set_color('#000000')
        axis[i, j].plot(averaged_run_data)

        legends_list = []

        while True:
            input_legend = input('Enter legend value('' to stop ): ')
            if input_legend == '':
                break
            else:
                legends_list.append(input_legend)
        if not legends_list:
            legends_list = ['Approx-Q', 'Episodic-Sarsa', 'Online-Sarsa']
        axis[i, j].legend(legends_list, loc='upper left', fontsize=font_size_labels)

figure.tight_layout()
plt.show()

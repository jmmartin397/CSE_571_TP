import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np


def calculate_convergence_episode(run_df, window_width, slope_window_width, slope_threshold):
    """Roughly calculates the episode at which each algorithm in each run
    converges.

    This is done by smoothing the data by applying a rolling average and then
    finding the first point at which the slope between points a given distance
    in front of and behind it falls below a threshold.
    """
    run_df = run_df.rolling(window_width).mean().dropna()
    slope_offset = slope_window_width // 2
    num_episodes = run_df.shape[0]
    first_idx = run_df.index[0]
    convergence_episodes = {alg_name: None for alg_name in run_df.columns}
    for idx in range(first_idx + slope_offset, num_episodes - slope_offset):
        first_point = idx - slope_offset
        second_point = idx + slope_offset
        for alg_name in run_df.columns:
            rise = run_df[alg_name][second_point] - run_df[alg_name][first_point]
            run = slope_window_width
            slope = rise / run
            if slope < slope_threshold and convergence_episodes[alg_name] is None:
                convergence_episodes[alg_name] = idx
    return convergence_episodes


def convergence_df(batch_name, **kwargs):
    run_dfs = get_runs(batch_name, min_val=-1000)
    ce = lambda run: calculate_convergence_episode(run, **kwargs)
    convergence_episodes = list(map(ce, run_dfs))
    ce_df = pd.DataFrame(convergence_episodes)
    return ce_df


def print_convergence_times_for_anova(batch_name, **kwargs):
    ce_df = convergence_df(batch_name, **kwargs)
    for col in ce_df.columns:
        times = list(ce_df[col])
        print('{}: {}'.format(col, times))


def print_convergence_values_for_anova(batch_name, num_to_average):
    """calculates the average score of the last 100 episodes in each run
    """
    run_dfs = get_runs(batch_name, -1000)
    values_lists = {col:[] for col in run_dfs[0].columns}
    for run in run_dfs:
        for col in run.columns:
            last_episodes = run[col][-num_to_average:]
            value = sum(last_episodes) / len(last_episodes)
            values_lists[col].append(value)
    for col in values_lists:
        print('{}: {}'.format(col, values_lists[col]))


def get_runs(batch_name, min_val=None, max_val=None):
    """Retrieves data and bounds outliers
    """
    run_dfs = evaluate.read_run_data(batch_name)
    #remove outliers
    if min_val is not None:
        for i in range(len(run_dfs)):
            run_dfs[i][run_dfs[i] < min_val] = min_val
    if max_val is not None:
        for i in range(len(run_dfs)):
            run_dfs[i][run_dfs[i] > max_val] = max_val
    return run_dfs



if __name__ == '__main__':
    # dump out data used in anova tests
    params = {
        'window_width': 200,
        'slope_window_width': 100,
        'slope_threshold': .1,
    }

    bn1 = 'all_agents_better_extractor_standard_params_mediumClassic_100_1000' #anova P-value: 0.2375
    print('MediumClass convergence times:')
    print_convergence_times_for_anova(bn1, **params)
    print('MediumClass convergence values:')
    print_convergence_values_for_anova(bn1, num_to_average=100)
    
    bn2 = 'all_agents_better_extractor_standard_params_random_layouts_100_1000'
    print('Randomized Layouts convergence times:')
    print_convergence_times_for_anova(bn2, **params)
    print('Randomized Layouts convergence values:')
    print_convergence_values_for_anova(bn2, num_to_average=100)

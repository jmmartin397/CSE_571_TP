import evaluate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import convergence

def plot_all_runs_rolling_average(batchname):
    run_dfs = evaluate.read_run_data(batchname)
    for i in range(10):
        run_dfs[i].rolling(100).mean().plot()
        plt.show()

def plot_run_with_slope_at_convergence(run):
    window_width = 200
    slope_window_width = 100
    slope_threshold = .1
    ce = convergence.calculate_convergence_episode(
        run, window_width, slope_window_width, slope_threshold)
    run = run.rolling(window_width).mean().dropna()
    run.plot()
    for alg_name in run.columns:
        if ce[alg_name] is not None:
            first_idx = ce[alg_name] - slope_window_width // 2
            second_idx = ce[alg_name] + slope_window_width // 2
            x_coords = [first_idx, second_idx]
            y_coords = [run[alg_name][first_idx], run[alg_name][second_idx]]
            print(x_coords, y_coords)
            plt.plot(x_coords, y_coords)
    print(ce)
    plt.show()
    return ce


def plot_comparison_of_hists(df1, title1, df2, title2, fig_title):
    fig, axes = plt.subplots(1,2)
    for idx, params in enumerate(zip(axes, [df1, df2], [title1, title2])):
        ax, df, title = params
        plt.sca(ax)
        plt.hist([df[col] for col in df.columns], label=df.columns)
        plt.title(title, fontsize=32)
        plt.xlabel('Episodes', fontsize=20)
        plt.legend(fontsize=16)
        plt.xlabel('Episodes', fontsize=24)
        plt.tick_params(left=False, labelleft=False)
        plt.xticks(fontsize=16)
    fig.suptitle(fig_title, fontsize=36)
    plt.show()


def plot_two_batches(bn1, bn2, title1, title2):
    fig, axes = plt.subplots(2,1)
    for idx, params in enumerate(zip(axes, [bn1, bn2], [title1, title2])):
        ax, bn, title = params
        plt.sca(ax)
        runs = convergence.get_runs(bn, -1000)
        averaged_run_data = sum(runs) / len(runs)
        averaged_run_data = averaged_run_data.rolling(100).mean()
        averaged_run_data.plot(ax=ax)
        plt.title(title, fontsize=32)
        if idx == 1:
            plt.xlabel('Episodes', fontsize=28)
        else:
            plt.tick_params(bottom=False, labelbottom=False)
        plt.legend(fontsize=24)
        plt.tick_params(left=False, labelleft=False)
        plt.xticks(fontsize=24)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Total Reward (averaged over all runs)', fontsize=28)
    plt.show()


def plot_data_set(batchname, title):
    runs = convergence.get_runs(batchname)
    averaged_run_data = sum(runs) / len(runs)
    averaged_run_data = averaged_run_data.rolling(1).mean()
    averaged_run_data.plot()
    plt.title(title, fontsize=32)
    plt.ylabel('Total Reward (averaged over all runs)', fontsize=28)
    plt.xlabel('Episodes', fontsize=28)
    plt.legend(fontsize=24)
    plt.yticks(fontsize=24)
    plt.xticks(fontsize=24)
    plt.show()


def plot_data_sets(batchnames, titles):
    fig, axes = plt.subplots(3,1)
    for idx, params in enumerate(zip(axes, batchnames, titles)):
        ax, bn, title = params
        plt.sca(ax)
        runs = convergence.get_runs(bn)
        averaged_run_data = sum(runs) / len(runs)
        averaged_run_data = averaged_run_data.rolling(1).mean()
        averaged_run_data.plot(ax=ax)
        ax.yaxis.tick_right()
        plt.title(title, fontsize=32)
        plt.yticks(fontsize=24)
        plt.ylim((-400, 1700))
        plt.xticks(fontsize=24)
        if idx == 0:
            plt.legend(fontsize=24)
        else:
            ax.get_legend().remove()
        if idx == 2:
            plt.xlabel('Episodes', fontsize=28)
        else:
            plt.tick_params(bottom=False, labelbottom=False)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Total Reward (averaged over all runs)', fontsize=28)
    plt.show()


if __name__ == '__main__':
    # create Fig. 1. (comparison of Lambda values)
    # bn = 'compare_lambda_better_extractor_alpha_1e-2_100_300'
    # plot_data_set(bn, 'Effect of Lambda Value')
    
    # create Fig. 2. (comparison of Alpha values)
    # bns = [
    #     'all_agents_better_extractor_alpha_1e-1_100_300',
    #     'all_agents_better_extractor_alpha_1e-2_100_300',
    #     'all_agents_better_extractor_alpha_1e-3_100_300',
    # ]
    # titles = [
    #     'Alpha = 0.1',
    #     'Alpha = 0.01',
    #     'Alpha = 0.001',
    # ]
    # plot_data_sets(bns, titles)

    # create Fig. 3. (comparison of training convergence)
    # bn2 = 'all_agents_better_extractor_standard_params_random_layouts_100_1000'
    # bn1 = 'all_agents_better_extractor_standard_params_mediumClassic_100_1000' #anova P-value: 0.2375
    # title1 = 'Medium Classic Layout'
    # title2 = 'Random Layout'
    # plot_two_batches(bn1, bn2, title1, title2)

    # create Fig. 4. (Time to convergence histograms)
    # params = {
    #     'window_width': 200,
    #     'slope_window_width': 100,
    #     'slope_threshold': .1,
    # }
    # bn2 = 'all_agents_better_extractor_standard_params_random_layouts_100_1000'
    # bn1 = 'all_agents_better_extractor_standard_params_mediumClassic_100_1000'
    # ce_df1 = convergence.convergence_df(bn1, **params)
    # ce_df2 = convergence.convergence_df(bn2, **params)
    # plot_comparison_of_hists(
    #     ce_df1,
    #     'MediumClassic Layout',
    #     ce_df2,
    #     'Random Layout',
    #     'Time to Convergence'
    # )
    pass

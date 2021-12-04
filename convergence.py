import evaluate
import matplotlib.pyplot as plt


def plot_rolling():
    bn = 'aprox_q_hunter_extractor_hunt_lay_varying_alpha_100_300'
    bn = 'all_agents_better_extractor_alpha_1e-1_100_300'
    run_dfs = evaluate.read_run_data(bn)

    for i in range(10):
        run_dfs[i].rolling(100).mean().plot()
        plt.show()


def calculate_convergence_episode(run_df, window_width, slope_window_width, slope_threshold):
    run_df = run_df.rolling(window_width).mean().dropna()
    # print(run_df)
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


def plot_run_with_slope_at_convergence(run):
    window_width = 100
    slope_window_width = 40
    slope_threshold = 5
    run = run_dfs[0]
    ce = calculate_convergence_episode(run, window_width, slope_window_width, slope_threshold)
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

if __name__ == '__main__':
    # plot_rolling()
    bn = 'all_agents_better_extractor_alpha_1e-1_100_300'
    # # bn = 'aprox_q_hunter_extractor_hunt_lay_alpha_1e-1_varying_gamma_100_1000'
    run_dfs = evaluate.read_run_data(bn)
    run = run_dfs[0]

    plot_run_with_slope_at_convergence(run)

    # window_width = 100
    # slope_window_width = 10
    # slope_threshold = 10
    # ce = calculate_convergence_episode(run, window_width, slope_window_width, slope_threshold)
    # print(ce)
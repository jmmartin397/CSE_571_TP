from pacman import ClassicGameRules
from qlearningAgents import ApproximateQAgent
from qlearningAgents import EpisodicSemiGradientSarsaAgent
from qlearningAgents import TrueOnlineLambdaSarsa
import textDisplay
import ghostAgents
import layout
from matplotlib import pyplot as plt
import tqdm
import pandas as pd
import os
import multiprocessing
from multiprocessing import Pool
import signal

RUN_DATA_DIRECTORY = './data'


def train_and_evaluate_worker(args):
    try:
        # unpack location-based params
        run_no, batch_name, agent_classes, common_params, agent_specific_params, env, num_episodes = args
        display = textDisplay.NullGraphics()
        rules = ClassicGameRules()  
        agents = dict()
        for agent_name, agent_class in agent_classes.items():
            agents[agent_name] = agent_class(**common_params, **agent_specific_params[agent_name])
        run = {agent_name: [] for agent_name in agents}
        for agent_name, agent in agents.items():
            for episode_no in range(num_episodes):
                #train game
                agent.switch_to_train_mode()
                game = rules.newGame(
                    layout=env['layout'],
                    pacmanAgent=agent,
                    ghostAgents=env['ghosts'],
                    display=display
                )
                game.run()
                #test game
                agent.switch_to_test_mode()
                game = rules.newGame(
                    layout=env['layout'],
                    pacmanAgent=agent,
                    ghostAgents=env['ghosts'],
                    display=display
                )
                game.run()
                run[agent_name].append(game.state.getScore())
        #convert to dataframe in order to save to csv easily
        run_df = pd.DataFrame(run)
        os.makedirs(os.path.join(RUN_DATA_DIRECTORY, batch_name), exist_ok=True)
        fname = 'run{}.csv'.format(run_no)
        save_path = os.path.join(RUN_DATA_DIRECTORY, batch_name, fname)
        run_df.to_csv(save_path, index=False)
    except KeyboardInterrupt:
        return



def train_and_evaluate_single_thread(run_indexes, *args):
    num_runs = len(run_indexes)
    for run_no in tqdm.tqdm(run_indexes):
        train_and_evaluate_worker(tuple([run_no, *args]))


def train_and_evaluate(run_indexes, batch_name, *args, proc_count=None):
    """ Train and evaluate agents using the train_and_evalaute_worker function.

    If proc_count is > 1, execute in parallel on proc_count processes.
    If proc_count is not given, use one less than total number of system threads (generally 2 * core count)
        This is to allow for responsive GUI

    Keyboard interrupt (Ctrl-C) will kill all threads.
    Will pick up where is left off if it was interrupted.
    """
    num_runs = len(run_indexes)
    if proc_count is None:
        # try to leave at least one thread free for GUI repsonsiveness
        proc_count = max(1, multiprocessing.cpu_count() - 1)
    
    pool = Pool(processes=proc_count)#, initializer=initializer)
    params = [tuple([run_no, batch_name, *args]) for run_no in run_indexes]
    desc = '{} processes'.format(proc_count)
    try:
        for _ in tqdm.tqdm(pool.imap_unordered(train_and_evaluate_worker, params), total=num_runs, desc=desc):
            pass
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print('ctrl-c pressed')
        pool.terminate()

    # calculate how many runs were completed (may have been interupted)
    max_run_idx = max(run_indexes)
    runs_left_to_go = get_batch_run_indexes_yet_to_go(batch_name, max_run_idx)
    num_runs_complted = num_runs - len(runs_left_to_go)
    print('Completed {} runs'.format(num_runs_complted))


def calculate_num_runs_completed(batch_name):
    return len(os.listdir(os.path.join(RUN_DATA_DIRECTORY, batch_name)))


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


def generate_graph(batch_name, params):
    runs = read_run_data(batch_name)
    averaged_run_data = average_over_runs(runs)
    averaged_run_data.plot()
    title_str = ' | '.join(['{}={}'.format(key, value) for key, value in params.items()])
    plt.title(title_str)
    plt.ylabel('Total reward on episode (avg over {} runs)'.format(calculate_num_runs_completed(batch_name)))
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()
    

def get_batch_run_indexes_yet_to_go(batch_name, num_runs):
    """ returns a list of the index numbers from 0 to num_runs -1 have yet to
    had data written to a run#.csv file where # in the run index.
    """
    fnames = os.listdir(os.path.join(RUN_DATA_DIRECTORY, batch_name))
    # fname format: run#.csv where # is an int
    #extract that int
    completed_run_indexes = [int(fname[3:].split('.')[0]) for fname in fnames]
    indexes_yet_to_go = [idx for idx in range(num_runs) if idx not in completed_run_indexes]
    indexes_yet_to_go.sort()
    return indexes_yet_to_go



if __name__ == '__main__':
    common_params = {
        'alpha': .1,
        'epsilon': .05,
        'gamma': .9,
    }
    agent_specific_params = {
        'approx-q': {'extractor': 'BetterExtractor'},
        'epi-sarsa': {'extractor': 'BetterExtractor'},
        'online-sarsa': {'extractor': 'BetterExtractor'},
    }
    agent_classes = {
        'approx-q': ApproximateQAgent,
        'epi-sarsa': EpisodicSemiGradientSarsaAgent,
        'online-sarsa': TrueOnlineLambdaSarsa
    }
    envs = {
        'standardPacman': {
            'layout': layout.getLayout('mediumClassic'),
            'ghosts': [ghostAgents.RandomGhost(i+1) for i in range(2)],
        },
        'trickyPacman': {
            'layout': layout.getLayout('trickyClassic'),
            'ghosts': [ghostAgents.RandomGhost(i+1) for i in range(4)],
        },
        'sparsePacman': {
            'layout': layout.getLayout('sparseMedClassic'),
            'ghosts': [ghostAgents.RandomGhost(i+1) for i in range(2)],
        },
    }
    num_runs = 100
    num_episodes = 300

    batch_name = 'all_agents_better_extractor_{}_{}'.format(num_runs, num_episodes)

    # a list of run indexes that have not yet been completed
    # (i.e. no run#.csv file for index # yet)
    run_indexes = get_batch_run_indexes_yet_to_go(batch_name, num_runs)


    # comment out this function call to just generate the graph
    train_and_evaluate(
        run_indexes,
        batch_name,
        agent_classes, 
        common_params, 
        agent_specific_params, 
        envs['standardPacman'],
        num_episodes,
        proc_count=None
    )
    
    generate_graph(batch_name, common_params)

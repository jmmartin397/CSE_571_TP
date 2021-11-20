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


def train_and_evaluate(agent_classes, common_params, agent_specific_params, env, num_runs, num_episodes, data_dir_name):
    display = textDisplay.NullGraphics()
    rules = ClassicGameRules()    
    #generate data
    for run_no in tqdm.tqdm(range(num_runs)):
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
        os.makedirs(data_dir_name, exist_ok=True)
        fname = 'run{}.csv'.format(run_no)
        save_path = os.path.join(data_dir_name, fname)
        run_df.to_csv(save_path, index=False)
        
def read_run_data(data_dir_name, num_runs):
    runs = []
    for run_no in range(num_runs):
        fname = 'run{}.csv'.format(run_no)
        path = os.path.join(data_dir_name, fname)
        run = pd.read_csv(path)
        runs.append(run)
    return runs

def average_over_runs(runs):
    return sum(runs) / len(runs)

def generate_graph(run_data, params, num_runs):
    run_data.plot()
    title_str = ' | '.join(['{}={}'.format(key, value) for key, value in params.items()])
    plt.title(title_str)
    plt.ylabel('Total reward on episode (avg over {} runs)'.format(num_runs))
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    common_params = {
        'alpha': 1e-5,
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

    data_dir_name = 'data/all_agents_better_extractor_{}_{}'.format(num_runs, num_episodes)

    train_and_evaluate(
        agent_classes, 
        common_params, 
        agent_specific_params, 
        envs['standardPacman'],
        num_runs,
        num_episodes,
        data_dir_name
    )

    runs = read_run_data(data_dir_name, num_runs)
    averaged_run_data = average_over_runs(runs)
    generate_graph(averaged_run_data, common_params, num_runs)

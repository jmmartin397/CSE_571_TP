from featureExtractors import SimpleExtractor
from game import Game
from pacman import ClassicGameRules
from qlearningAgents import ApproximateQAgent
import textDisplay
import graphicsDisplay
import ghostAgents
import layout

def evaluate(agents, hyper_params, envs, num_training, eval_interval, num_eval):
    print('evaluating {} agents in {} environments using parameters:'.format(
        len(agents), len(envs)))
    print(hyper_params)
    print('each agent will train for {} games in each env.'.format(num_training))
    print('agents will be evaluated (0 alpha, 0 epsilon) every {} games for {} games.'.format(
        eval_interval, num_eval))
    print('evaluation scores are averages.')
    display = textDisplay.NullGraphics()
    evalDisplay = graphicsDisplay.PacmanGraphics()
    rules = ClassicGameRules()
    # print()
    # print('Format:')
    # print('{}:{}:{}/{}'.format('env', 'agent', '#gamesTrained', '#gamesToGo'))
    # print()
    for env in envs:
        for agent in agents:
            scores = []
            for game_no in range(num_training):
                game = rules.newGame(
                    layout=envs[env]['layout'],
                    pacmanAgent=agents[agent],
                    ghostAgents=envs[env]['ghosts'],
                    display=display
                )
                game.run()
                if game_no % eval_interval == 0:
                    eval_scores = []
                    old_alpha = agents[agent].alpha
                    agents[agent].alpha = 0
                    old_epsilon = agents[agent].epsilon
                    agents[agent].epsilon = 0
                    for eval_no in range(num_eval):
                        game = rules.newGame(
                            layout=envs[env]['layout'],
                            pacmanAgent=agents[agent],
                            ghostAgents=envs[env]['ghosts'],
                            display=display
                        )
                        game.run()
                        eval_scores.append(game.state.getScore())
                    scores.append(sum(eval_scores) / len(eval_scores))
                    agents[agent].epsilon = old_epsilon
                    agents[agent].alpha - old_alpha
                    print('{}/{}/{}: {}'.format(env, agent, game_no, scores[-1]))
            print('{}/{}/{}: {}'.format(env, agent, game_no, scores))


if __name__ == '__main__':
    agent_hyper_params = {
        'alpha': .05,
        'epsilon': .05,
        'gamma': .8,
    }
    agents = {
        'approx-q': ApproximateQAgent(**agent_hyper_params, extractor='SimpleExtractor'),
    }
    envs = {
        'standardPacman': {
            'layout': layout.getLayout('mediumClassic'),
            'ghosts': [ghostAgents.RandomGhost(i+1) for i in range(2)],
        }
    }

    # train for 200 games, stopping to evaluate for 5 games every 20 games
    #results = evaluate(agents, agent_hyper_params, envs, 200, 20, 5)
    #print(results)

    # after every game, evaluate for 5 games
    results = evaluate(agents, agent_hyper_params, envs, 20, 1, 5)

from pacman import ClassicGameRules
from qlearningAgents import ApproximateQAgent
from qlearningAgents import EpisodicSemiGradientSarsaAgent
import textDisplay
import graphicsDisplay
import ghostAgents
import layout
from matplotlib import pyplot as plt
import tqdm



def average_reward_on_episode(agent_classes, common_params, agent_specific_params, env, num_runs, num_episodes):
    runs = []
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
                # agent.switch_to_test_mode()
                # game = rules.newGame(
                #     layout=env['layout'],
                #     pacmanAgent=agent,
                #     ghostAgents=env['ghosts'],
                #     display=display
                # )
                # game.run()
                run[agent_name].append(game.state.getScore())
        runs.append(run)
    #reformat data
    results = dict()
    for agent_name in agent_classes:
        results[agent_name] = []
        for episode in range(num_episodes):
            total_episode_score = sum(run[agent_name][episode] for run in runs)
            average_episode_score = total_episode_score / num_runs
            results[agent_name].append(average_episode_score)
    return results

def generate_graph(result_data, num_runs, num_episodes):
    episode_nums = list(range(1,num_episodes+1))
    for agent_name, avg_episode_scores in result_data.items():
        plt.plot(episode_nums, avg_episode_scores, label=agent_name)

    plt.ylabel('Total reward on episode (avg over {} runs)'.format(num_runs))
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()
    

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
    common_params = {
        'alpha': .01,
        'epsilon': .05,
        'gamma': .9,
    }
    agent_specific_params = {
        'approx-q-b': {'extractor': 'BetterExtractor'},
        'approx-q-s': {'extractor': 'SimpleExtractor'},
        'epi-sarsa-s': {'extractor': 'SimpleExtractor'},
        'epi-sarsa-b': {'extractor': 'BetterExtractor'},
    }
    agent_classes = {
        'approx-q-b': ApproximateQAgent,
        # 'approx-q-s': ApproximateQAgent,
        #'epi-sarsa': EpisodicSemiGradientSarsaAgent,
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

    num_runs = 1
    num_episodes = 1000

    results = average_reward_on_episode(
        agent_classes, 
        common_params, 
        agent_specific_params, 
        envs['sparsePacman'],
        num_runs,
        num_episodes
    )
    print(results)
    # results = {'approx-q': [243.3, 1216.1, 969.9, 1193.9, 1214.8, 932.7, 1223.8, 1321.6, 1319.9, 996.5],
    #            'epi-sarsa': [141.9, 1083.0, 1096.9, 1215.8, 1206.8, 825.5, 1204.9, 1087.9, 1224.6, 920.4]}

    generate_graph(results, num_runs, num_episodes)

    # train for 200 games, stopping to evaluate for 5 games every 20 games
    #results = evaluate(agents, agent_hyper_params, envs, 200, 20, 5)
    #print(results)

    # after every game, evaluate for 5 games
    # results = evaluate(agents, agent_hyper_params, envs, 20, 1, 5)


# results = {'approx-q': [1047.3, 1050.3, 1187.95, 1138.65, 1221.35, 1196.7, 1125.05, 1322.65, 1331.15, 1135.6, 1143.25, 1201.15, 1202.05, 1147.55, 1153.75, 1068.55, 1191.2, 
#                         1266.9, 1070.85, 1272.4, 1129.6, 1206.1, 1126.9, 1042.5, 1335.15, 1010.3, 1191.25, 1138.0, 1091.15, 1188.4, 1195.3, 1320.95, 1015.7, 1194.9, 1209.8, 
#                         1269.4, 1253.55, 1351.85, 1357.1, 1279.5, 1271.45, 1236.8, 1181.95, 1233.5, 1206.35, 1202.05, 1267.4, 1079.5, 1300.8, 1118.1, 1180.45, 1319.35, 1259.65, 
#                         1272.95, 1274.2, 1153.95, 1125.75, 1202.9, 1329.75, 1325.3, 1330.9, 1267.3, 1280.65, 1267.5, 1066.25, 1186.95, 1253.55, 1106.65, 1193.5, 1258.8, 1145.7, 
#                         1269.05, 1287.95, 1280.15, 1260.75, 1231.4, 1143.7, 1194.95, 1286.55, 1196.0, 1127.8, 1176.9, 1327.1, 1320.75, 1142.65, 1274.15, 1193.05, 1333.4, 1158.7, 
#                         1133.85, 1237.4, 1059.85, 1234.85, 1210.65, 1349.55, 1209.9, 1211.75, 1055.55, 1254.25, 1330.65], 'epi-sarsa': [1111.05, 1247.9, 1267.35, 1322.05, 1086.15, 
#                         973.9, 1210.2, 1271.3, 1319.65, 1149.1, 1075.5, 1275.2, 1324.7, 1200.05, 1205.7, 1254.7, 1254.35, 1280.65, 1212.7, 1274.6, 1322.85, 1257.2, 1147.2, 1257.5,
#                          1221.65, 1153.95, 1005.1, 1198.3, 1217.75, 1274.75, 1276.45, 1197.25, 1158.85, 1212.75, 1155.25, 1090.65, 1194.0, 1319.7, 1256.65, 1100.25, 1154.2, 1221.3, 
#                          1135.75, 1116.7, 1185.65, 1279.35, 1095.1, 1278.8, 1252.1, 1250.35, 1211.45, 1246.35, 1174.1, 1194.1, 1268.35, 1335.5, 1080.05, 1143.9, 1188.4, 1267.05, 
#                          1069.75, 1127.55, 1265.65, 1268.25, 1159.7, 1325.5, 1134.65, 1328.7, 1196.55, 1269.1, 1208.85, 1199.15, 1241.0, 1209.5, 1344.0, 1183.15, 1319.5, 1083.85, 
#                          1268.6, 1330.4, 1185.95, 1315.4, 1212.6, 1326.95, 1237.5, 1138.6, 1266.75, 1342.65, 1203.0, 1073.45, 1023.65, 1266.7, 1340.7, 1146.7, 1174.0, 1117.3, 1174.4, 
#                          1073.6, 1241.05, 1204.55]}
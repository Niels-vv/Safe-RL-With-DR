import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.realpath(__file__))

def show_agent_results(dir):
    results_path = f'{PATH}/../{dir}'

    i = 1
    results_file = f'{results_path}/results_{i}.csv'
    while(os.path.isfile(results_file)):
        results = pd.read_csv(results_file)
        rewards = results.loc[:, "Reward"]
        learning_rewards = []
        eval_rewards = []
        for r in range(len(results)):
            if r % 15 == 0 and r > 0:
                eval_rewards.append(rewards[r])
            else:
                learning_rewards.append(rewards[r])
        show_plot("Episode", "Reward", learning_rewards, f'Learning results for agent {i}')
        show_plot("Episode", "Reward", eval_rewards, f'Evaluation results for agent {i}')
        i += 1
        results_file = f'{results_path}/results_{i}.csv' 

def show_plot(x_name, y_name, y, title):
    plt.figure(2)
    plt.clf()        
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.plot(y, '-b')
    plt.show()

if __name__ == "__main__":
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    show_agent_results(dir)
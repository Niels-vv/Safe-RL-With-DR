import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__))

def show_agent_results(results_file, name):
    results = pd.read_csv(results_file)
    rewards = results.loc[:, "Reward"]
    learning_rewards = []
    eval_rewards = []
    for r in range(len(results)):
        if r % 15 == 0 and r > 0:
            eval_rewards.append(rewards[r])
        else:
            learning_rewards.append(rewards[r])
    learning_period = 30
    eval_period = 5
    learning_average = get_average(learning_rewards, learning_period)
    eval_average = get_average(eval_rewards, eval_period)
    show_plot("Episode", "Reward", learning_rewards, learning_average, learning_period, f'Learning results for {name}')
    show_plot("Episode", "Reward", eval_rewards, eval_average, eval_period, f'Evaluation results for {name}')

def get_average(values, period):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def compare_base_agents(dir):
    results_path = f'{PATH}/../{dir}'

    i = 1
    results_file = f'{results_path}/results_{i}.csv'
    while(os.path.isfile(results_file)):
        name = f'base agent {i}'
        show_agent_results(results_file, name)
        i += 1
        results_file = f'{results_path}/results_{i}.csv' 

def compare_base_and_vae_agents(dir):
    results_path = f'{PATH}/../{dir}'
    base_results_file = f'{results_path}/results_1.csv'
    base_name = "base agent"
    vae_results_file = f'{results_path}/results_vae_1.csv'
    vae_name = "agent using vae"
    show_agent_results(base_results_file, base_name)
    show_agent_results(vae_results_file, vae_name)

def show_plot(x_name, y_name, rewards, average, period, title):
    plt.figure(2)
    plt.clf()        
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.plot(rewards, '-b', label = "Rewards per episode")
    plt.plot(average, '-g', label = f'Average reward per {period} episodes')
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    compare_base_and_vae_agents(dir)
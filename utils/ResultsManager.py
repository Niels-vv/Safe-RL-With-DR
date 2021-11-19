import os, torch, math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.DataManager import DataManager
from vae.VAE import VAE, VaeManager
import seaborn as sns

PATH = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentPerformance:

    @staticmethod
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
        learning_average = AgentPerformance.get_average(learning_rewards, learning_period)
        eval_average = AgentPerformance.get_average(eval_rewards, eval_period)
        AgentPerformance.show_plot("Episode", "Reward", learning_rewards, learning_average, learning_period, f'Learning results for {name}')
        AgentPerformance.show_plot("Episode", "Reward", eval_rewards, eval_average, eval_period, f'Evaluation results for {name}')

    @staticmethod
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

    @staticmethod
    def compare_base_agents(dir):
        results_path = f'{PATH}/../{dir}'

        i = 1
        results_file = f'{results_path}/results_{i}.csv'
        while(os.path.isfile(results_file)):
            name = f'base agent {i}'
            AgentPerformance.show_agent_results(results_file, name)
            i += 1
            results_file = f'{results_path}/results_{i}.csv' 

    @staticmethod
    def compare_base_and_vae_agents(dir):
        results_path = f'{PATH}/../{dir}'
        base_results_file = f'{results_path}/results_1.csv'
        base_name = "base agent"
        vae_results_file = f'{results_path}/results_vae_1.csv'
        vae_name = "agent using vae"
        AgentPerformance.show_agent_results(base_results_file, base_name)
        AgentPerformance.show_agent_results(vae_results_file, vae_name)

    @staticmethod
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

class AEAnalysis:
    def reduced_features_correlation_matrix(ae, obs_dir):
        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()
        num_features = math.sqrt(ae.latent_space)

        reduced_states = []
        states = []

        i = 1
        max_num_states = 1
        for index, row in obs.iterrows():
            state = row.reshape(32,32)
            state = torch.from_numpy(row).to(device).float()
            state_red = ae.vae_model.state_dim_reduction(state).detach().cpu().numpy()
            reduced_states.append(state_red)
            states.append(state.cpu().numpy())

            i += 1
            if i > max_num_states: break

        data = {}
        df_ae = pd.DataFrame(data = reduced_states[0],columns=list(range(reduced_states[0].shape[1])))
        corr_matrix = df_ae.corr()
        sns.heatmap(corr_matrix, annot=False)
        plt.show()

    @staticmethod
    def get_component(vae_dir, vae_name):
        checkpoint = DataManager.get_network(vae_dir, vae_name, device)
        vae_model = VAE(in_channels = 0, latent_dim = checkpoint['latent_space']).to(device) #todo in_channels weghalen
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        vae_optimizer = None
        vae_model.eval()
        return VaeManager(vae_model, vae_optimizer, checkpoint['obs_file'],checkpoint['batch_size'], checkpoint['latent_space'], checkpoint['vae_lr'])



def show_base_vae_comparison():
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    AgentPerformance.compare_base_and_vae_agents(dir)

def get_ae():
    ae_dir = "env_pysc2/results_vae/MoveToBeacon"
    ae_name = "vae.pt"
    return AEAnalysis.get_component(ae_dir, ae_name)

def show_new_reduced_features_correlation():
    ae = get_ae()
    obs_dir = "/content/drive/MyDrive/Thesis/Code/PySC2/Observations/MoveToBeacon"
    AEAnalysis.reduced_features_correlation_matrix(ae, obs_dir)

if __name__ == "__main__":
    pass
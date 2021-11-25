import os, torch, math, random
import torch.nn as nn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utils.DataManager import DataManager
from vae.VAE import VAE, VaeManager
import seaborn as sns

PATH = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class methods showing performance and comparison of different types of agents (e.g. base agent, or agent using autoencoder)
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
    def compare_base_and_ae_agents(dir):
        results_path = f'{PATH}/../{dir}'
        base_results_file = f'{results_path}/results_1.csv'
        base_name = "base agent"
        ae_results_file = f'{results_path}/results_vae_1.csv'
        ae_name = "agent using vae"
        AgentPerformance.show_agent_results(base_results_file, base_name)
        AgentPerformance.show_agent_results(ae_results_file, ae_name)

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

# Class methods showing analyses of autoencoder, e.g. visualizing feature maps and showing correlation matrix
class AEAnalysis:
    def reduced_features_correlation_matrix(ae, obs_dir):
        data_manager = DataManager(observation_sub_dir = obs_dir)
        print("Retrieving observations...")
        obs = data_manager.get_observations()

        reduced_states = []
        states = []

        i = 1
        matrix_num = 1
        max_num_states = 60000
        print("Retrieving states and reduced states...")
        
        for index, row in obs.iterrows():
            row = row.to_numpy()
            state = row.reshape(32,32)
            state = torch.from_numpy(state).to(device).float()
            state_red = ae.state_dim_reduction(state).detach().cpu().numpy().flatten()
            reduced_states.append(state_red)
            states.append(state.detach().cpu().numpy().flatten())

            i += 1
            if i > max_num_states:
                states = pd.DataFrame(states, columns=list(range(states[0].shape[0])))
                reduced_states = pd.DataFrame(reduced_states, columns=list(range(reduced_states[0].shape[0])))

                print(f'Calculating correlation matrix {matrix_num}...')
                df_cor = pd.concat([states, reduced_states], axis=1, keys=['df1', 'df2']).corr().loc['df2', 'df1']
                fig, ax = plt.subplots(figsize=(30,8))
                sns.heatmap(df_cor, annot=False, ax=ax)
                plt.savefig(f'corr_matrix_reduced_{matrix_num}.png')
                plt.show()

                # Reset for calculating new matrix for next batch of observations (due to memory limits)
                matrix_num += 1
                i = 1
                reduced_states = []
                states = []
                print("Retrieving states and reduced states...")
        print("Done calculating and storing correlation matrices")

    @staticmethod
    def visualize_feature_maps(model, obs_dir):
        def get_rectangle(area):
            for width in range(int(math.ceil(math.sqrt(area))), 1, -1):
                if (area % width == 0): break
            return width, int(area/width)

        activation = {}
        # Get feature map data
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cpu()
            return hook

        print("Retrieving observations...")
        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()

        # Get conv layers from encoder
        conv_layers = []
        for layer in model.encoder:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
                
        print("Getting feature maps and storing images...")
        max_num_states = 2
        state_num = 0 # Number of states we have analysed
        jump = 10996 # Take every jumpth state, as to not get similar states
        for index, row in obs.iterrows():
            if state_num >= max_num_states: break
            if index % jump != 0: continue
            row = row.to_numpy()
            state = row.reshape(32,32)

            # Store original state as image
            norm = plt.Normalize(vmin=state.min(), vmax=state.max())
            fig, axarr = plt.subplots(1)
            axarr.imshow(norm(state))
            plt.savefig(f'State_{index+1}_original.png')

            state = torch.from_numpy(state).to(device).float()
            for i in range(len(conv_layers)):
                conv_layers[i].register_forward_hook(get_activation(f'conv{i}'))
            model.state_dim_reduction(state)

            print(f'Creating feature maps for state {state_num+1} / {max_num_states}...')
            for i in range(len(conv_layers)):
                image_name = f'State_{index+1}_Layer_{i+1}_Feature_Map.png'

                # Conv layer only has 1 channel/feature map
                if (activation[f'conv{i}'].shape[1] == 1):
                    act = activation[f'conv{i}'].squeeze()
                    fig, axarr = plt.subplots(1)
                    axarr.imshow(act)
                    plt.savefig(image_name)
                    continue

                # Conv layer only has > 1 channels/feature maps: show them in a rectangle grid
                act = activation[f'conv{i}'].squeeze()
                width, length = get_rectangle(act.size(0))
                fig, axarr = plt.subplots(width, length)
                r = axarr.shape[0]
                c = axarr.shape[1]
                for idxi in range(r):
                    for idxj in range(c):
                        if idxi*c+idxj >= len(act): break
                        axarr[idxi, idxj].imshow(act[idxi*c+idxj])
                plt.savefig(image_name)

            state_num +=1

    @staticmethod
    def get_component(vae_dir, vae_name):
        checkpoint = DataManager.get_network(vae_dir, vae_name, device)
        vae_model = VAE(in_channels = 0, latent_dim = checkpoint['latent_space']).to(device) #todo in_channels weghalen
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        vae_optimizer = None
        vae_model.eval()
        return VaeManager(vae_model, vae_optimizer, checkpoint['obs_file'],checkpoint['batch_size'], checkpoint['latent_space'], checkpoint['vae_lr'])

    @staticmethod
    def get_ae():
        ae_dir = "env_pysc2/results_vae/MoveToBeacon"
        ae_name = "vae.pt"
        return AEAnalysis.get_component(ae_dir, ae_name)

### Results analysis methods

# Compare the results (in rewards/episode) of a base agent with an agent using an autoencoder
def show_base_ae_comparison():
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    AgentPerformance.compare_base_and_ae_agents(dir)

# Show a heatmap of the correlation matrix between original state features and reduced state features (reduced by an autoencoder)
def show_reduced_features_correlation():
    ae = AEAnalysis.get_ae().vae_model
    obs_dir = "/content/drive/MyDrive/Thesis/Code/PySC2/Observations/MoveToBeacon"
    AEAnalysis.reduced_features_correlation_matrix(ae, obs_dir)

# Visualize the feature map of the encoder of an autoencoder
def show_feature_map_ae():
    ae = AEAnalysis.get_ae().vae_model
    obs_dir = "/content/drive/MyDrive/Thesis/Code/PySC2/Observations/MoveToBeacon"
    AEAnalysis.visualize_feature_maps(ae, obs_dir)

def test():
    obs_dir = "/content/drive/MyDrive/Thesis/Code/PySC2/Observations/MoveToBeacon"
    data_manager = DataManager(observation_sub_dir = obs_dir)
    obs = data_manager.get_observations()
    max_index = (0,-1,0) # row, column, value
    min_index = (0,1024,0)
    indices_min = []
    indices_max = []
    print("Calculating...")
    for index, row in obs.iterrows():
        row = row.to_numpy()
        min_column = (0,1024,0)
        max_column = (0,-1,0)

        for c in range(len(row)):
            if row[c] > 0:
                if c <= min_column[1]:
                    min_column = (index,c,row[c])
                if c >= max_column[1]:
                    max_column = (index, c, row[c])
        if min_column[1] <= min_index[1]: min_index = min_column
        if max_column[1] >= max_index[1]: max_index = max_column
        indices_min.append(min_column)
        indices_max.append(max_column)
    print(f'max_index: {max_index}')
    print(f'min_index: {min_index}')
    print("Max indices")
    print(indices_max)
    print("Min indices")
    print(indices_min)

if __name__ == "__main__":
    #show_reduced_features_correlation()
    show_feature_map_ae()
    #test()
    
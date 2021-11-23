import os, torch, math
import torch.nn as nn
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
        obs = data_manager.get_observations()

        reduced_states = []
        states = []

        i = 1
        max_num_states = 1
        for index, row in obs.iterrows():
            state = row.reshape(32,32)
            state = torch.from_numpy(row).to(device).float()
            state_red = ae.state_dim_reduction(state).detach().cpu().numpy().flatten()
            reduced_states.append(state_red)
            states.append(state.cpu().numpy().flatten())

            i += 1
            if i > max_num_states: break

        states = pd.DataFrame(states, columns=list(range(states[0].shape[1])))
        reduced_states = pd.DataFrame(reduced_states, columns=list(range(reduced_states[0].shape[1])))
        df_cor = pd.concat([states, reduced_states], axis=1, keys=['df1', 'df2']).corr().loc['df2', 'df1']
        sns.heatmap(df_cor, annot=False)
        plt.show()

    @staticmethod
    def visualize_feature_maps_backup(model, obs_dir):
        def visualize_layers(layer_results):
            for num_layer in range(len(layer_results)):
                plt.figure(figsize=(30, 30))
                layer_viz = layer_results[num_layer][0, :, :, :]
                layer_viz = layer_viz.data
                for i, filter in enumerate(layer_viz):
                    if i == 64: # we will visualize only 8x8 blocks from each layer
                        break
                    plt.subplot(8, 8, i + 1)
                    plt.imshow(filter, cmap='gray')
                    plt.axis("off")
                plt.close()

        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()

        conv_layers = []
        for layer in model.encoder.layer:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
        
        i = 1
        max_num_states = 1
        for index, row in obs.iterrows():
            state = row.reshape(32,32)
            plt.imshow(state)

            state = torch.from_numpy(row).to(device).float().unsqueeze(0).unsqueeze(0)
            results = [conv_layers[0](state)]
            for i in range(1, len(conv_layers)):
                # pass the result from the last layer to the next layer
                results.append(conv_layers[i](results[-1]))

            visualize_layers(results)

            i += 1
            if i > max_num_states: break

    #TODO remove one or the other visualisation method
    @staticmethod
    def visualize_feature_maps_backup(model, obs_dir):
        def visualize_layers(layer_results):
            i = 1
            for num_layer in range(len(layer_results)):
                print(f'Feature maps for conv layer {i}')
                plt.figure(figsize=(30, 30))
                layer_viz = layer_results[num_layer][0, :, :, :]
                layer_viz = layer_viz.data
                for i, filter in enumerate(layer_viz):
                    if i == 64: # we will visualize only 8x8 blocks from each layer
                        break
                    plt.subplot(8, 8, i + 1)
                    plt.imshow(filter, cmap='gray')
                    plt.axis("off")
                plt.close()
                i += 1

        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()

        conv_layers = []
        for layer in model.encoder.layer:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
        
        i = 1
        max_num_states = 1
        for index, row in obs.iterrows():
            state = row.reshape(32,32)
            plt.imshow(state)

            state = torch.from_numpy(row).to(device).float().unsqueeze(0).unsqueeze(0)
            results = [conv_layers[0](state)]
            for i in range(1, len(conv_layers)):
                # pass the result from the last layer to the next layer
                results.append(conv_layers[i](results[-1]))

            visualize_layers(results)

            i += 1
            if i > max_num_states: break

    @staticmethod
    def visualize_feature_maps(model, obs_dir):
        def normalize_output(img):
            img = img - img.min()
            img = img / img.max()
            return img

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()

        conv_layers = []
        for layer in model.encoder.layer:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
        
        i = 1
        max_num_states = 1
        for index, row in obs.iterrows():
            state = row.reshape(32,32)
            plt.imshow(state)

            state = torch.from_numpy(row).to(device).float()
            for i in range(len(conv_layers)):
                conv_layers[i].register_forward_hook(get_activation(f'conv{i}'))
            result = model.state_dim_reduction(state).detach().cpu().numpy()

            for i in range(len(conv_layers)):
                print(f'Feature maps for conv layer {i}')
                act = activation[f'conv{i}'].squeeze()
                fig, axarr = plt.subplots(act.size(0))
                for idx in range(act.size(0)):
                    axarr[idx].imshow(act[idx])

            i += 1
            if i > max_num_states: break

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

if __name__ == "__main__":
    pass
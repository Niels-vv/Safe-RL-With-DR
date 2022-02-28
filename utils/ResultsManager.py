import os, torch, math, random
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utils.DataManager import DataManager
from vae.VAE import VAE, VaeManager
import seaborn as sns
from math import sqrt

PATH = os.path.dirname(os.path.realpath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class methods showing performance and comparison of different types of agents (e.g. base agent, or agent using autoencoder)
class AgentPerformance:

    @staticmethod
    def show_agent_results(results_file, name, store_img_filename):
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
        AgentPerformance.show_plot("Episode", "Reward", learning_rewards, learning_average, learning_period, f'Learning results for {name}', store_img_filename)
        #AgentPerformance.show_plot("Episode", "Reward", eval_rewards, eval_average, eval_period, f'Evaluation results for {name}')

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
        store_filename_base = "Base_agent_results.png"
        base_name = "base agent"

        ae_results_file = f'{results_path}/results_vae_1.csv'
        ae_name = "Pre-trained autoencoder agent"
        store_filename_ae = "Pretrained_autoencoder_agent_results.png"
        AgentPerformance.show_agent_results(base_results_file, base_name, store_filename_base)
        AgentPerformance.show_agent_results(ae_results_file, ae_name, store_filename_ae)
    
    @staticmethod
    def compare_online_ae_and_deepmdp_agents(dir):
        results_path = f'{PATH}/../{dir}'
        ae_results_file = f'{results_path}/results_ae_online_1.csv'
        store_filename_ae = "Online_autoencoder_agent_results.png"
        ae_name = "Online trained autoencoder agent"

        deepmdp_results_file = f'{results_path}/results_deepmdp_1.csv'
        deepmdp_name = "DeepMDP agent"
        store_filename_deepmdp = "Deepmdp_agent_results.png"
        AgentPerformance.show_agent_results(ae_results_file, ae_name, store_filename_ae)
        AgentPerformance.show_agent_results(deepmdp_results_file, deepmdp_name, store_filename_deepmdp)

    @staticmethod
    def show_pca_agent_results(dir):
        results_path = f'{PATH}/../{dir}'
        pca_results_file = f'{results_path}/results_pca_scalar.csv'
        store_filename_pca = "PCA_with_scalar_agent_results.png"
        pca_name = "PCA agent"
        AgentPerformance.show_agent_results(pca_results_file, pca_name, store_filename_pca)

    @staticmethod
    def show_plot(x_name, y_name, rewards, average, period, title, filename):
        plt.figure(2)
        plt.clf()        
        plt.title(title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.plot(rewards, '-b', label = "Rewards per episode")
        plt.plot(average, '-g', label = f'Average reward per {period} episodes')
        plt.legend(loc="upper left")
        plt.savefig(filename)
        plt.show()

# Class methods showing analyses of PCA
class PCAAnalysis:    
    @staticmethod
    def show_state_representation_pca(obs_dir, pca, dim_name, recon_name):
        def show_state_representation(obs):
            # Get state in latent space and save image
            latent_repr = pca.state_dim_reduction(obs)
            state = torch.reshape(latent_repr, (int(sqrt(pca.latent_space)), int(sqrt(pca.latent_space)))).detach().cpu().numpy()
            norm = plt.Normalize(vmin=state.min(), vmax=state.max())
            fig, axarr = plt.subplots(1)
            axarr.imshow(norm(state))
            plt.savefig(dim_name)

        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()

        jump = 10996 
        for index, row in obs.iterrows():
            if index % jump != 0 or index == 0: continue
            row = row.to_numpy()
            state = row.reshape(32,32)

            state = torch.tensor(state, dtype=torch.float, device=device)
            break
        
        show_state_representation(state)

        # Store original state as image
        norm = plt.Normalize(vmin=state.min(), vmax=state.max())
        fig, axarr = plt.subplots(1)
        axarr.imshow(norm(state.cpu().numpy()))
        plt.savefig(f'State_check_original.png')

        # TODO remove test
        plt.imshow(state.cpu().numpy())
        plt.savefig("state_check_version2.png")

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
        model.eval()
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
    def visualize_filters(model):
        # for i, layer in enumerate(model.encoder[0], start = 1):
        for i, layer in enumerate(model.encoder, start = 1):
            if type(layer) == nn.Conv2d:
                image_name = f'Filters_Layer_{i+1}_Feature_Map.png'
                filters = layer.weight.numpy()
                plt.imshow(filters[0, ...])
                plt.savefig(image_name)

                # f_min, f_max = filters.min(), filters.max()
                # filters = (filters - f_min) / (f_max - f_min)

                # # Conv layer only has 1 channel/feature map
                # if (len(filters) == 1):
                #     fig, axarr = plt.subplots(1)
                #     axarr.imshow(filters[0])
                #     plt.savefig(image_name)
                #     continue

                # # Conv layer only has > 1 channels/feature maps: show them in a rectangle grid
                # width, length = get_rectangle(filters.size(0))
                # fig, axarr = plt.subplots(width, length)
                # r = axarr.shape[0]
                # c = axarr.shape[1]
                # for idxi in range(r):
                #     for idxj in range(c):
                #         if idxi*c+idxj >= len(filters): break
                #         axarr[idxi, idxj].imshow(filters[idxi*c+idxj])
                # plt.savefig(image_name)

    # Find pixel values for which ae feature maps are activated the most
    # https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    @staticmethod
    def activation_image(model):
        model.eval()
        activation = {}
        # Get feature map data
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = torch.tensor(output,requires_grad=True, device = device)
            return hook

        # Get conv layers from encoder
        conv_layers = []
        for layer in model.encoder:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)

        for i in range(len(conv_layers)):
            conv_layers[i].register_forward_hook(get_activation(f'conv{i}'))
            for filter in conv_layers[i].weight.shape(0):
                img = get_image(i, filter)
                plt.imsave(f'activation_image_layer_{i}_filter_{filter}.jpg', np.clip(img, 0, 1))
        
        # Get image that activates given filter of given layer the most
        def get_image(layer, filter):
            width = 32
            height = 32
            colour_channels = 1
            img = np.uint8(np.random.uniform(150, 180, (width, height, colour_channels)))/255
            img = torch.from_numpy(img).to(device)
            img.requires_grad_()

            opt_steps = 20
            optimizer = torch.optim.Adam([img], lr=0.1, weight_decay=1e-6)
            for _ in range(opt_steps):
                optimizer.zero_grad()
                model(img)
                loss = -activation[f'conv{layer}'][filter].mean()
                loss.backward()
                optimizer.step()
            return img.detach().cpu().numpy()


    # GRAD-LAM
    @staticmethod
    def get_heatmap(model, obs_dir):
        model.eval()
        activation = []
        gradients = None
        # Get feature map data
        def get_activation():
            def hook(model, input, output):
                activation.append(output.detach().cpu())
                output.register_hook(activations_hook)
            return hook
        #Get gradients
        def activations_hook(self,grad):
            gradients = grad

        print("Retrieving observations...")
        data_manager = DataManager(observation_sub_dir = obs_dir)
        obs = data_manager.get_observations()

        # Get conv layers from encoder
        conv_layers = []
        for layer in model.encoder:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
        last_conv = conv_layers[-1]
                
        # Forward pass with state index 10996
        print("Getting heatmap and storing images...")
        state = None
        for index, row in obs.iterrows():
            if index == 10996:
                row = row.to_numpy()
                state = row.reshape(32,32)
                break
        state = torch.from_numpy(state).to(device).float()
        last_conv.register_forward_hook(get_activation())
        pred = model.state_dim_reduction(state).argmax(dim=1)
        pred[:, 386].backward()

        # Use gradients and activations to create heatmap
        acts = activation[0].squeeze()
        gradients = gradients.detach().cpu()





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

def get_rectangle(area):
    for width in range(int(math.ceil(math.sqrt(area))), 1, -1):
        if (area % width == 0): break
    return width, int(area/width)

### Results analysis methods

# Compare the results (in rewards/episode) of a base agent with an agent using a pre-trained autoencoder
def show_base_ae_comparison():
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    AgentPerformance.compare_base_and_ae_agents(dir)

# Compare the results (in rewards/episode) of an agent using an online trained autoencoder with a DeepMDP agent
def show_online_ae_deepmdp_comparison():
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    results_dir = f'{PATH}/../{dir}'
    if not os.path.isfile(f'{results_dir}/results_deepmdp_1.csv'): # DeepMDP results have not yet been combined: combine them to new csv file
        i = 1
        filenames = []
        while os.path.isfile(f'{results_dir}/results_deepmdp_1-{i}.csv'):
            filenames.append(f'{results_dir}/results_deepmdp_1-{i}.csv')
            i += 1
        combined_csv = pd.concat([pd.read_csv(f) for f in filenames ])
        combined_csv.to_csv(f'{results_dir}/results_deepmdp_1.csv', index=False, encoding='utf-8-sig')
    AgentPerformance.compare_online_ae_and_deepmdp_agents(dir)

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
    
# Visualize the filters of the encoder of an autoencoder
def show_filters_ae():
    ae = AEAnalysis.get_ae().vae_model
    AEAnalysis.visualize_filters(ae)

def most_activation_image_ae():
    ae = AEAnalysis.get_ae().vae_model
    AEAnalysis.activation_image(ae)

def show_pca_agent_results():
    dir = "env_pysc2/results/dqn/MoveToBeacon"
    AgentPerformance.show_pca_agent_results(dir)

def pca_analyses():
    obs_dir = "/content/drive/MyDrive/Thesis/Code/PySC2/Observations/MoveToBeacon"
    # Analysis of PCA with scalar
    dim_name = "pca_with_scalar_latent_state.png"
    recon_name = "pca_with_scalar_reconstructed_state.png"
    pca = DataManager.get_component(f'env_pysc2/results_pca/MoveToBeacon',"pca1_with_scalar.pt") # TODO check of naamgeving nog klopt
    PCAAnalysis.show_state_representation_pca(obs_dir, pca, dim_name, recon_name)

    # Analysis of PCA without scalar
    dim_name = "pca_without_scalar_latent_state.png"
    recon_name = "pca_without_scalar_reconstructed_state.png"
    pca = DataManager.get_component(f'env_pysc2/results_pca/MoveToBeacon',"pca2_no_scalar.pt") # TODO check of naamgeving nog klopt
    PCAAnalysis.show_state_representation_pca(obs_dir, pca, dim_name, recon_name)

def show_epsilon_decay():
    epsilons = 0.01 / np.logspace(-2, 0, 100000, endpoint=False) - 0.01
    epsilons = epsilons * (1.0 - 0.1) + 0.1
    epsilons = [x for i,x in enumerate(epsilons) if i % 239 == 0 ]
    plt.figure(2)
    plt.clf()        
    plt.title("Epsilon decay")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.plot(epsilons, '-b', label = "Epsilon value per episode")
    plt.legend(loc="upper right")
    plt.savefig("Epsilon_decay.png")
    plt.show()


if __name__ == "__main__":
    pca_analyses()
    

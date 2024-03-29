import os, torch, math, time,copy,psutil
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utils.DataManager import DataManager
from utils.MaxActivation import act_max
from autoencoder.ae import AE
from env_atari.models import ae_encoder, ae_decoder
import seaborn as sns
from math import sqrt

from utils.DeepDream import deep_dream_static_image

PATH = os.path.dirname(os.path.realpath(__file__))
process = psutil.Process(os.getpid())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class methods showing performance and comparison of different types of agents (e.g. base agent, or agent using autoencoder)
class AgentPerformance:

    @staticmethod
    def show_agent_results(results_file, name, store_img_filename,yrange):
        i = 1
        rewards = []
        while os.path.isfile(f'{results_file}_{i}.csv'):
            r = pd.read_csv(f'{results_file}_{i}.csv')
            rewards += r.loc[:, "Reward"].tolist()
            i += 1

        learning_period = 30
        learning_average = AgentPerformance.get_average(rewards, learning_period)
        AgentPerformance.show_plot("Episode", "Reward", rewards, learning_average, learning_period, f'Training results for {name}', store_img_filename, yrange)
        
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
    def show_plot(x_name, y_name, rewards, average, period, title, filename,yrange):
        plt.figure(2)
        plt.clf()        
        plt.title(title)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.plot(rewards, '-b', label = "Rewards per episode")
        plt.plot(average, '-g', label = f'Average reward per {period} episodes')
        plt.ylim([yrange[0],yrange[1]])
        plt.legend(loc="lower right")
        plt.savefig(filename)
        plt.show()

    def show_average_plots(rewards, title, yrange, filename):
        plt.figure(2)
        plt.clf()        
        plt.title(title)
        plt.xlabel("episode")
        plt.ylabel("reward average per 30 episodes")
        plt.plot(rewards[0], '-b', label = "Baseline")
        plt.plot(rewards[1], '-g', label = "PCA")
        plt.plot(rewards[2], '-r', label = "Pre-trained AE")
        plt.plot(rewards[3], '-c', label = "Online trained AE")
        if len(rewards) == 5:
            plt.plot(rewards[4], '-m', label = "DeepMDP")
        plt.ylim([yrange[0],yrange[1]])
        plt.legend(loc="upper left")
        plt.savefig(filename)
        plt.show()

# Class methods showing analyses of PCA
class PCAAnalysis:    
    @staticmethod
    def show_state_representation_pca(obs, indices, pca, file_name):
        def show_state_representation(state, filename):
            # Get state in latent space and save image
            latent_repr = pca.state_dim_reduction(state)
            norm = plt.Normalize(vmin=latent_repr.min(), vmax=latent_repr.max())
            fig, axarr = plt.subplots(1)
            axarr.imshow(norm(latent_repr))
            plt.savefig(filename)

        for state_index in indices:
            state = obs[state_index]
            filename = f'{file_name}_{state_index+1}.png'
            show_state_representation(state, filename)

# Class methods showing analyses of autoencoder, e.g. visualizing feature maps and showing correlation matrix
class AEAnalysis:
    def reduced_features_correlation_matrix(ae, obs, features, original_space, latent_space):
        def create_plot(filename, data, fig_size):
            fig, ax = plt.subplots(figsize=fig_size)
            sns.heatmap(data, vmin=-1, vmax=1, annot=False, ax=ax)
            plt.savefig(filename)
            plt.show()

        # Get correlation matrix of single latent feature
        def get_cor_of_latent_feature(corr_matrix, feature):
            cor_latent_feature = corr_matrix[:, feature].reshape(original_space[0], original_space[1])
            cor_latent_feature = pd.DataFrame(cor_latent_feature,columns=list(range(cor_latent_feature[0].shape[0])))
            create_plot(f'corr_matrix_latent_feature_{feature+1}.png', cor_latent_feature, (10,5))
     
        # Show placement of latent feature in its latent space
        def get_grid_for_latent_feature(feature, grid_rows, grid_columns):
            grid = np.ones(grid_rows *grid_columns)
            grid[feature] = 0
            grid = grid.reshape(grid_rows, grid_columns) 
            plt.figure(figsize=(5, 5))
            plt.imshow(grid, cmap='gray')
            plt.savefig(f'grid_latent_feature_{feature+1}.jpg')
            plt.show()

        reduced_states = []

        i = 0
        max_num_states = min(60000, len(obs))
        print("Retrieving states and reduced states...")
        
        for state in obs:
            state = torch.from_numpy(state).to(device).unsqueeze(0).unsqueeze(0)
            state_red = ae.state_dim_reduction(state).detach().cpu().numpy().flatten()
            reduced_states.append(state_red)
            i += 1
            if i >= max_num_states:
                break
        print(f'Retrieved reduced')
        reduced_states = np.array(reduced_states)
        # i=0
        # for state in obs:
        #     states.append(state.flatten())
        #     i+=1
        #     if i >= max_num_states:
        #         break
        # del obs        
        obs = np.array([ob.flatten() for ob in obs][:max_num_states])

        print(f'Calculating correlation matrix...')
        df_cor = np.concatenate((obs,reduced_states),axis=1) # Concat so we can run np.corrcoef on the (obs + reduced states) x (obs + reduced states) matrxix
        df_cor = np.corrcoef(df_cor,rowvar=False)[:obs.shape[1], obs.shape[1]:] # Take the obs x reduced states matrix from the returned (obs + reduced states) x (obs + reduced states) coeff matrix
        print("creating plot")
        create_plot(f'corr_matrix_reduced.png', df_cor,(8,30))

        # Show correlation for single latent feature
        for feature in features:
            get_cor_of_latent_feature(df_cor, feature)
            get_grid_for_latent_feature(feature, grid_rows = latent_space[0], grid_columns = latent_space[1])

        print("Done calculating and storing correlation matrices")

    @staticmethod
    def visualize_feature_maps(model, obs, state_indices):
        activation = {}
        # Get feature map data
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach().cpu()
            return hook

        # Get conv layers from encoder
        conv_layers = []
        for layer in model.encoder:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
                
        print("Getting feature maps and storing images...")
        for index in state_indices:
            state = obs[index]

            # Store original state as image
            norm = plt.Normalize(vmin=state.min(), vmax=state.max())
            fig, axarr = plt.subplots(1)
            axarr.imshow(norm(state))
            plt.savefig(f'State_{index+1}_original.png')

            state = torch.from_numpy(state).to(device).unsqueeze(0).unsqueeze(0)
            for i in range(len(conv_layers)):
                conv_layers[i].register_forward_hook(get_activation(f'conv{i}'))
            
            output = model.state_dim_reduction(state).squeeze().detach().cpu().numpy()
            # Store ae encodcer output state as image
            norm = plt.Normalize(vmin=output.min(), vmax=output.max())
            fig, axarr = plt.subplots(1)
            axarr.imshow(norm(output))
            plt.savefig(f'State_{index+1}_Encoder_Output.png')

            print(f'Creating feature maps for state...')
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

    @staticmethod
    def visualize_filters(model):
        # for i, layer in enumerate(model.encoder[0], start = 1):
        for i, layer in enumerate(model.encoder, start = 1):
            if type(layer) == nn.Conv2d:
                image_name = f'Filters_Layer_{i+1}_Feature_Map.png'
                filters = layer.weight.detach().cpu().numpy()
                #plt.imshow(filters[0, ...])
                #plt.savefig(image_name)

                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min)

                # Conv layer only has 1 channel/feature map
                if (len(filters) == 1):
                      fig, axarr = plt.subplots(1)
                      axarr.imshow(filters[0,0,:,:], cmap='gray')
                      plt.savefig(image_name)
                      plt.show()
                      continue

                # # Conv layer only has > 1 channels/feature maps: show them in a rectangle grid
                width, length = get_rectangle(filters.shape[0])
                plt.figure(figsize=(20, 17))
                for i, filter in enumerate(filters):
                    plt.subplot(width, length, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                    plt.imshow(filter[0,:,:], cmap='gray')
                    plt.savefig(image_name)
                plt.show()

    # Find pixel values for which ae feature maps are activated the most
    # https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    @staticmethod
    def activation_image(model, shape, deepdream=True):
        #model.eval()
        activation = {}

        # Get feature map data
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.requires_grad_(True)
            return hook

        # Get image that activates given filter of given layer the most
        def get_image(layer, filter):
            width = 32
            height = 32
            colour_channels = 1
            #img = torch.randn(1,colour_channels,height, width, requires_grad=True,device=device)   
            img = np.random.uniform(low=0.0, high=1.0, size=(1,1,32,32)).astype(np.float32)
            #img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = torch.from_numpy(img).to(device).requires_grad_(True)
            opt_steps = 50
            optimizer = torch.optim.Adam([img], lr=0.1)
            for _ in range(opt_steps):
                model.zero_grad()
                optimizer.zero_grad()
                img.retain_grad()
                
                model(img)
                loss = -activation[f'conv{layer}'][:,filter,:,:].mean()
                # loss = torch.nn.MSELoss(reduction='mean')(activation[f'conv{layer}'], torch.zeros_like(activation[f'conv{layer}']))
                loss.backward()
                optimizer.step()
            
            return img.detach().cpu().squeeze().numpy()
        
        model = model.encoder
        # Get conv layers from encoder
        conv_layers = []
        for layer in model:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
        

        for i in range(len(conv_layers)):
            conv_layers[i].register_forward_hook(get_activation(f'conv{i}'))
            if deepdream:
                img = deep_dream_static_image(model, i, activation, (shape[0],shape[1],1))
                plt.imshow(img)
                plt.savefig(f'DeepDream_activation_image_layer_{i}.jpg')
                continue
            for filter in range(conv_layers[i].weight.shape[0]):
                print(f'Getting image for filter {filter+1} of conv layer {i+1}.')
                img = np.random.uniform(low=0.0, high=1.0, size=(1,1,shape[0],shape[1])).astype(np.float32)
                #img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
                img = torch.from_numpy(img).to(device).requires_grad_(True)
            
                #img = get_image(i,filter)
                img = act_max(model, img, activation, f'conv{i}', filter)
                plt.imshow(img)
                plt.savefig(f'activation_image_layer_{i}_filter_{filter}.jpg')
                break
            break

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
    def get_component(ae_dir, ae_name):
        checkpoint = DataManager.get_network(ae_dir, ae_name, device)
        ae_model = AE(ae_encoder, ae_decoder).to(device)
        ae_model.load_state_dict(checkpoint['model_state_dict'])
        ae_model.eval()
        return ae_model

def get_rectangle(area):
    for width in range(int(math.ceil(math.sqrt(area))), 1, -1):
        if (area % width == 0): break
    return width, int(area/width)

### Results analysis methods

# Show average results for all agents
def show_average_results_all(results_path, yrange, env_name):
    filename = f'Average_results_agents_{env_name}.png'
    title = f'Training results for all agents in {env_name}'
    result_names = ["Results", "Results_pca", "Results_ae", "Results_online_ae"]
    if env_name == "pysc2": result_names.append("Results_deepmdp")
    results = []
    for res in result_names:
        i = 1
        rewards = []
        while os.path.isfile(f'{results_path}/{res}_{i}.csv'):
            r = pd.read_csv(f'{results_path}/{res}_{i}.csv')
            rewards += r.loc[:, "Reward"].tolist()
            i += 1
        learning_period = 30
        results.append(AgentPerformance.get_average(rewards, learning_period))
    AgentPerformance.show_average_plots(results, title, yrange, filename)

# Show results for baseline agent.
def show_base_results(results_path, yrange):
    base_results_file = f'{results_path}/Results'
    store_filename_base = "Base_agent_results.png"
    base_name = "Baseline agent"
    AgentPerformance.show_agent_results(base_results_file, base_name, store_filename_base, yrange)

# Show results for pre-trained ae agent.
def show_pretrained_ae_results(results_path, yrange):
    ae_results_file = f'{results_path}/Results_ae'
    ae_name = "Pre-trained autoencoder agent"
    store_filename_ae = "Pretrained_autoencoder_agent_results.png"
    AgentPerformance.show_agent_results(ae_results_file, ae_name, store_filename_ae, yrange)

# Show results for non-pretrained (online trained) ae agent
def show_online_ae_results(results_path, yrange):
    ae_results_file = f'{results_path}/Results_online_ae'
    store_filename_ae = "Online_autoencoder_agent_results.png"
    ae_name = "Online trained autoencoder agent"
    AgentPerformance.show_agent_results(ae_results_file, ae_name, store_filename_ae, yrange)

def show_pca_agent_results(results_path, yrange):
    pca_results_file = f'{results_path}/Results_pca'
    store_filename_pca = "PCA_agent_results.png"
    pca_name = "PCA agent"
    AgentPerformance.show_agent_results(pca_results_file, pca_name, store_filename_pca, yrange)

# Show results for deepmdp agent
def show_deepmdp_agent_results(results_path, yrange):
    deepmdp_results_file = f'{results_path}/Results_deepmdp'
    deepmdp_name = "DeepMDP agent"
    store_filename_deepmdp = "Deepmdp_agent_results.png"
    AgentPerformance.show_agent_results(deepmdp_results_file, deepmdp_name, store_filename_deepmdp, yrange)

# Show a heatmap of the correlation matrix between original state features and reduced state features (reduced by an autoencoder)
def show_reduced_features_correlation(ae, obs, features, original_shape, latent_shape):
    AEAnalysis.reduced_features_correlation_matrix(ae, obs, features, original_shape, latent_shape)

# Visualize the feature map of the encoder of an autoencoder
def show_feature_map_ae(ae, obs, states):
    AEAnalysis.visualize_feature_maps(ae, obs, states)
    
# Visualize the filters of the encoder of an autoencoder
def show_filters_ae(ae):
    AEAnalysis.visualize_filters(ae)

def most_activation_image_ae(ae, shape, deepdream=True):
    AEAnalysis.activation_image(ae,shape, deepdream)

def pca_analyses(obs, state_indices, pca_path, pca_name):
    pca = DataManager.get_component(pca_path,pca_name)
    filename = "PCA_latent_representation"
    PCAAnalysis.show_state_representation_pca(obs, state_indices, pca, filename)

def show_epsilon_decay_pysc2():
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

def show_epsilon_decay_pong():
    epsilons = 0.01 / np.logspace(-2, 0, 100000, endpoint=False) - 0.01
    epsilons = epsilons * (1.0 - 0.1) + 0.1
    plt.figure(2)
    plt.clf()        
    plt.title("Epsilon decay")
    plt.xlabel("Steps")
    plt.ylabel("Epsilon")
    plt.plot(epsilons, '-b', label = "Epsilon value per step")
    plt.legend(loc="upper right")
    plt.savefig("Epsilon_decay_pong.png")
    plt.show()


if __name__ == "__main__":
    env_name = "pong"

    if env_name == "pysc2":
        results_dir = "env_pysc2/results/dqn/MoveToBeacon"
        yrange = (0,30)

        ae_dir = "env_pysc2/results_ae/MoveToBeacon"
        ae_name = "vae.pt"
        ae = AEAnalysis.get_component(ae_dir, ae_name)

        obs_dir = "drive/MyDrive/Thesis/Code/PySC2/Observations/MoveToBeacon"
        data_manager = DataManager(observation_sub_dir = obs_dir)
        data_manager.obs_file = f'{data_manager.observations_path}/Observations.npy'
        print("Retrieving observations...")
        #obs = data_manager.get_observations()

        original_shape = (32,32)
        latent_shape = (16,16)
        features = [67, 119, 203]
        states = [0,10966]


    elif env_name == "pong":
        results_dir = "env_atari/results/dqn/PongNoFrameskip-v4"
        yrange = (-21,21)

        ae_dir = "env_atari/results_ae/PongNoFrameskip-v4"
        ae_name = "ae.pt"
        ae = AEAnalysis.get_component(ae_dir, ae_name)

        pca_dir = "env_atari/results_pca/PongNoFrameskip-v4"
        pca_name = "pca.pt"

        obs_dir = "../drive/MyDrive/Thesis/Code/Atari/PongNoFrameskip-v4/Observations"
        data_manager = DataManager(observation_sub_dir = obs_dir)
        data_manager.obs_file = f'{data_manager.observations_path}/Observations_corr.npy'
        print("Retrieving observations...")
        #obs = data_manager.get_observations()   

        original_shape = (84,84)
        latent_shape = (42,42)
        features = [888, 903, 917]
        states = [10000,15000]
    
    results_path = f'{PATH}/../{results_dir}'

    print("Average results for all agents")
    show_average_results_all(results_path, yrange, env_name)
    #print("Baseline results")
    #show_base_results(results_path, yrange)
    #print("AE results")
    #show_pretrained_ae_results(results_path, yrange)
    # print("AE results")
    # show_online_ae_results(results_path, yrange)
    # print("PCA agent results")
    # show_pca_agent_results(results_path, yrange)
    # if env_name == "pysc2":
    #     print("DeepMDP agent results")
    #     show_deepmdp_agent_results(results_path, yrange)

    # print("PCA analyses")
    # pca_analyses(obs,states,results_dir,pca_name)

    #print("Correlation matrix")
    #show_reduced_features_correlation(ae, obs, features, original_shape, latent_shape)
    #print("feature maps")
    #show_feature_map_ae(ae, obs, states)
    #print("Filters")
    #show_filters_ae(ae)
    #print("activation")
    #most_activation_image_ae(ae, original_shape,deepdream=True)
    

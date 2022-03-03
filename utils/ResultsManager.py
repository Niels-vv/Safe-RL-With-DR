import os, torch, math, random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
from utils.DataManager import DataManager
from utils.MaxActivation import act_max
from vae.VAE import VAE, VaeManager
import seaborn as sns
from math import sqrt

from torch.functional import F #TODO weghalen

from utils.DeepDream import deep_dream_static_image

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
        def create_plot(filename, data, fig_size):
            fig, ax = plt.subplots(figsize=fig_size)
            sns.heatmap(data, vmin=-1, vmax=1, annot=False, ax=ax)
            plt.savefig(filename)
            plt.show()

        # Get correlation matrix of single latent feature
        def get_cor_of_latent_feature(corr_matrix, feature):
            cor_latent_feature = corr_matrix.iloc[feature].to_numpy().reshape(32,32)
            cor_latent_feature = pd.DataFrame(cor_latent_feature,columns=list(range(cor_latent_feature[0].shape[0])))
            create_plot(f'corr_matrix_latent_feature_{feature+1}.png', cor_latent_feature, (10,5))
     
        # Show placement of latent feature in its original 16x16 grid
        def get_grid_for_latent_feature(feature, grid_rows, grid_columns):
            grid = np.ones(grid_rows *grid_columns)
            grid[feature] = 0
            grid = grid.reshape(grid_rows, grid_columns) 
            plt.figure(figsize=(5, 5))
            plt.imshow(grid, cmap='gray')
            plt.savefig(f'grid_latent_feature_{feature+1}.jpg')
            plt.show()

        data_manager = DataManager(observation_sub_dir = obs_dir)
        print("Retrieving observations...")
        obs = data_manager.get_observations()

        reduced_states = []
        states = []

        i = 1
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

                print(f'Calculating correlation matrix...')
                df_cor = pd.concat([states, reduced_states], axis=1, keys=['df1', 'df2']).corr().loc['df2', 'df1']
                create_plot(f'corr_matrix_reduced.png', df_cor,(30,8))

                # Show correlation for single latent feature
                features = [67, 119, 203]
                for feature in features:
                    get_cor_of_latent_feature(df_cor, feature)
                    get_grid_for_latent_feature(feature, grid_rows = 16, grid_columns = 16)
                break

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
    def activation_image(model):
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
            img = torch.randn(1,colour_channels,height, width, requires_grad=True,device=device)   

            opt_steps = 50
            optimizer = torch.optim.Adam([img], lr=0.1, weight_decay=1e-6)
            for _ in range(opt_steps):
                model.zero_grad()
                optimizer.zero_grad()
                img.retain_grad()
                
                model(img)
                loss = -activation[f'conv{layer}'][:,filter,:,:].mean()
                # loss = torch.nn.MSELoss(reduction='mean')(activation[f'conv{layer}'], torch.zeros_like(activation[f'conv{layer}']))
                a = img.clone()
                loss.backward()
                optimizer.step()
                b = img.clone()
                print(img.grad)
                print(img.grad_fn)
                print(torch.equal(a.data, b.data))
            
            return img.detach().cpu().squeeze().numpy()
        
        model = model.encoder
        # Get conv layers from encoder
        conv_layers = []
        for layer in model:
            if type(layer) == nn.Conv2d:
                conv_layers.append(layer)
        

        for i in range(len(conv_layers)):
            conv_layers[i].register_forward_hook(get_activation(f'conv{i}'))
            #deep_dream_static_image(model, 0, activation)
            #deep_dream_static_image(model, 1, activation)
            for filter in range(conv_layers[i].weight.shape[0]):
                print(f'Getting image for filter {filter+1} of conv layer {i+1}.')
                initial_img = torch.randn(1, 32, 32, requires_grad=True).to(device)
                initial_img = initial_img.unsqueeze(0)
                img = get_image(i,filter)
                #img = act_max(model, activation, f'conv{i}', filter)
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
    def get_component(vae_dir, vae_name):
        checkpoint = DataManager.get_network(vae_dir, vae_name, device)
        vae_model = VAE(in_channels = 0, latent_dim = checkpoint['latent_space']).to(device) #todo in_channels weghalen
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        vae_optimizer = None
        #vae_model.eval()
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
    ae = AEAnalysis.get_ae().vae_model.to(device)
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

class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self):
        
        super().__init__()
        # initialize weights with random numbers
        weights = torch.randn(1,1,32, 32, requires_grad=True,device=device)  
        # make weights torch parameters
        self.weights = nn.Parameter(weights)        
        
    def forward(self, X):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        return X(self.weights)

def img_train():
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.requires_grad_(True)
        return hook
    ae = AEAnalysis.get_ae().vae_model.encoder.to(device)
    img_model = Model()
    optimizer = torch.optim.Adam(img_model.parameters(), lr=0.1, weight_decay=1e-6)
    a = img_model.weights.clone()
    y = torch.randn(1,1,32, 32, requires_grad=True,device=device)

    conv_layers = []
    for layer in ae:
        if type(layer) == nn.Conv2d:
            conv_layers.append(layer)
    conv_layers[0].register_forward_hook(get_activation(f'conv{0}'))

    img_model.weights.retain_grad()
    preds = img_model(ae)

    loss = -activation[f'conv{0}'][0][0].mean()
    print(loss)
    print(activation[f'conv{0}'][0].shape)
    print(activation[f'conv{0}'].shape)
    
    #loss = F.mse_loss(preds, y).sqrt()
    loss.backward()
    print(img_model.weights.grad)
    print(img_model.weights.grad[0][0][0][0])
    print(img_model.weights.grad[0][0][10][8])
    print(img_model.weights.grad[0][0][6][9])
    print(img_model.weights.grad[0][0][2][1])
    optimizer.step()
    print(img_model.weights.grad)
    print(img_model.weights.grad[0][0][0][0])
    print(img_model.weights.grad[0][0][10][8])
    print(img_model.weights.grad[0][0][6][9])
    print(img_model.weights.grad[0][0][2][1])
    optimizer.zero_grad()

    print(img_model.weights.grad)
    print(img_model.weights.grad[0][0][0][0])
    print(img_model.weights.grad[0][0][10][8])
    print(img_model.weights.grad[0][0][6][9])
    print(img_model.weights.grad[0][0][2][1])
    

    b = img_model.weights.clone()
    print(torch.equal(a.data,b.data))


if __name__ == "__main__":
    #img_train()
    most_activation_image_ae()
    
    # a = torch.randn(5,requires_grad=True).to(device)
    # print(a.grad_fn)
    # a.retain_grad()
    # a.backward(torch.ones(5).to(device))
    # print(a.grad)
    # print(a.grad_fn)
    

import sys, csv, torch, math
import pandas as pd
import numpy as np
from dqn.dqn import Agent as DQNAgent
from utils.DataManager import DataManager
from principal_component_analysis.pca import PCACompression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCAAgent(DQNAgent):
    def __init__(self, env, dqn_config, device, max_episodes, data_manager, mlp, conv_last, encoder, deepmdp, train, pca_config):
        pca_component = get_component(pca_config)
        latent_space = pca_component.latent_space
        print(f'PCA latent space: {latent_space}')
        print(f'PCA scalar: {pca_component.use_scalar}')
        super(PCAAgent, self).__init__(env, dqn_config, device, max_episodes, data_manager, mlp, conv_last, encoder, deepmdp, train)

        self.dim_reduction_component = pca_component
        self.reduce_dim = True
        self.pca = True

    def run_agent(self):
        super(PCAAgent, self).run_agent()

def get_component(config):
    return DataManager.get_component(config['file_path'],config['file_name'])

def train_pca(data_manager, config, get_statistics=True):
    # Get demo trajectory from observations file
    i = 1
    if (data_manager.create_dim_red_results_dirs(i)): # Get obs file (consisting of 40.000x4x84x84 obs)
        print("Retreiving observations...")
        observation_trace = data_manager.get_observations()
        observation_trace = observation_trace.reshape(observation_trace.shape[0] * observation_trace.shape[1], observation_trace.shape[2]*observation_trace.shape[3]) # Reshape to 1 channel if there are multiple ones like with atari and flatten images
    print(f'Observations shape:{observation_trace.shape}')

    # Train PCA on observation traces
    # Create initial PCA and get statistics on latent space
    pca_component = PCACompression(config['scalar'], config['latent_space'])
    pca_component.create_pca(observation_trace, get_statistics)
    statistics = pca_component.get_pca_dimension_info()
    latent_space = config['latent_space']
    print(f'Latent space {latent_space} info: {statistics[latent_space-1]}')

    # Create final PCA
    pca_component.update_pca()

    # Free up memory before storing PCA
    pca_component.df = None
    pca_component.pcaStatistic = None

    # Store PCA object as file
    data_manager.store_dim_reduction_component(pca_component, config['file_name'])
    print(f'Trained PCA on latent space {latent_space}')

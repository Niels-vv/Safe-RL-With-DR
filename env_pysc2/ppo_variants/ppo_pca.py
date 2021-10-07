import sys, csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from env_pysc2.ppo_variants.ppo_base import AgentLoop as Agent
from utils.DataManager import DataManager

class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name):
        if not train_component:
            super(AgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name, self.get_latent_space())

        self.train_component = train_component
        self.reduce_dim = not train_component
        self.pca = True

    def run_agent(self):
        if self.train_component:
            self.train_pca()
        else:
            self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results_pca/{self.map}')
            self.dim_reduction_component = self.data_manager.get_dim_reduction_component("pca.pt")
            super(AgentLoop, self).run_agent()

    def train_pca(self):
        # Get demo trajectory from observations file
        self.data_manager = DataManager(observation_sub_dir = f'env_pysc2/observations/{self.map}', results_sub_dir = f'env_pysc2/results_pca/{self.map}')
        obs = self.data_manager.get_observations()

        # Create initial PCA and get statistics on latent space
        pca_component = PCACompression()
        pca_component.create_pca(obs)
        print(pca_component.get_pca_dimension_info())

        # Set latent space and create final PCA
        latent_space = 100 # TODO
        pca_component.update_pca(obs, latent_space)

        # Store PCA object as file
        self.data_manager.store_dim_reduction_component(pca_component, "pca.pt")
        print(f'Trained PCA on latent space {latent_space}')

    def get_latent_space(self):
        self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results_pca/{self.map}')
        self.dim_reduction_component = self.data_manager.get_dim_reduction_component("pca.pt")
        return self.dim_reduction_component.latent_space

class PCACompression:
    def __init__(self):
        self.fileNames = []
        self.pca_main = None
        self.pcaStatistic = PCA()
        self.scaler = StandardScaler()
        self.latent_space = None

    def create_pca(self, observations):
        self.scaler.fit(observations)
        df = self.scaler.transform(observations)
        self.pca_main.fit(df)
        self.pcaStatistic.fit(df)
        
    def update_pca(self, observations, dimensions):
        self.latent_space = dimensions
        df = self.scaler.transform(observations)
        self.pca_main = PCA(n_components=dimensions)
        self.pca_main.fit(df)

    def state_dim_reduction(self, observation):
        obs = self.scaler.transform([observation.cpu().numpy()])
        return self.pca_main.transform(obs)[0]

    def get_pca_dimension_info(self):
        return np.cumsum(self.pcaStatistic.explained_variance_ratio_) 
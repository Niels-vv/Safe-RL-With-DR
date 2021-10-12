import sys, csv, torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from env_pysc2.ppo_variants.ppo_base import AgentLoop as Agent
from utils.DataManager import DataManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy):
        if not train_component: # We're not training PCA, but using it in PPO
            pca_component = self.get_component(map_name)
            latent_space = pca_component.latent_space
            super(AgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name, load_policy, latent_space, pca_component)
        else:
            self.map = map_name
        self.train_component = train_component
        self.reduce_dim = not train_component
        self.pca = True

    def run_agent(self):
        if self.train_component:
            self.train_pca()
        else:
            super(AgentLoop, self).run_agent()

    def get_component(self, map):
        return DataManager.get_component(f'env_pysc2/results_pca/{map}',"pca.pt")

    def train_pca(self):
        # Get demo trajectory from observations file
        self.data_manager = DataManager(observation_sub_dir = f'env_pysc2/observations/{self.map}', results_sub_dir = f'env_pysc2/results_pca/{self.map}')
        obs = self.data_manager.get_observations()
        print(obs.shape)

        # Create initial PCA and get statistics on latent space
        pca_component = PCACompression()
        pca_component.create_pca(obs)
        statistics = pca_component.get_pca_dimension_info()
        

        # Set latent space and create final PCA
        latent_space = 100 # TODO
        for i in range(len(statistics)):
            if statistics[i] > 0.90: #TODO magic nr
                latent_space = i + 1
                break
        print(latent_space)
        print(len(statistics)) # TODO klopt niet? pakt aantal rows ipv columns als max laten space wtf
        pca_component.update_pca(obs, latent_space)

        # Store PCA object as file
        self.data_manager.store_dim_reduction_component(pca_component, "pca.pt")
        print(f'Trained PCA on latent space {latent_space}')

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
        self.pcaStatistic.fit(df)
        
    def update_pca(self, observations, dimensions):
        self.latent_space = dimensions
        df = self.scaler.transform(observations)
        self.pca_main = PCA(n_components=dimensions)
        self.pca_main.fit(df)

    def state_dim_reduction(self, observation):
        obs = self.scaler.transform([observation.cpu().numpy()])
        return torch.tensor(self.pca_main.transform(obs)[0], dtype=torch.float, device=device)

    def get_pca_dimension_info(self):
        return np.cumsum(self.pcaStatistic.explained_variance_ratio_) 
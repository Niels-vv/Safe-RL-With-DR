import sys, csv, torch, math
import pandas as pd
import numpy as np
from env_pysc2.ppo_variants.ppo_base import AgentLoop as PPOAgent
from env_pysc2.dqn_variants.dqn_base import AgentLoop as DQNAgent
from utils.DataManager import DataManager
from pca.PCA import PCACompression

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgentLoop(DQNAgent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy):
        if not train_component: # We're not training PCA, but using it in PPO
            pca_component = get_component(map_name)
            latent_space = pca_component.latent_space
            super(DQNAgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name, load_policy, latent_space, pca_component)
        else:
            self.map = map_name
        self.train_component = train_component
        self.reduce_dim = not train_component
        self.pca = True

    def run_agent(self):
        if self.train_component:
            train_pca(self.map)
        else:
            super(DQNAgentLoop, self).run_agent()

class PPOAgentLoop(PPOAgent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy):
        if not train_component: # We're not training PCA, but using it in PPO
            pca_component = get_component(map_name)
            latent_space = pca_component.latent_space
            super(PPOAgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name, load_policy, latent_space, pca_component)
        else:
            self.map = map_name
        self.train_component = train_component
        self.reduce_dim = not train_component
        self.pca = True

    def run_agent(self):
        if self.train_component:
            train_pca(self.map)
        else:
            super(PPOAgentLoop, self).run_agent()

def get_component(map):
    return DataManager.get_component(f'env_pysc2/results_pca/{map}',"pca.pt")

def train_pca(map):
    # Get demo trajectory from observations file
    data_manager = DataManager(observation_sub_dir = f'/content/drive/MyDrive/Thesis/Code/PySC2/Observations/{map}', results_sub_dir = f'env_pysc2/results_pca/{map}')
    print("Retrieving observations...")
    obs = data_manager.get_observations()
    print(f'Observations shape:{obs.shape}')

    # Create initial PCA and get statistics on latent space
    pca_component = PCACompression()
    pca_component.create_pca(obs)
    statistics = pca_component.get_pca_dimension_info()

    # Set latent space and create final PCA
    for i in range(len(statistics)):
        if statistics[i] > 0.95 and isSquare(i+1): #TODO magic nr
            latent_space = i + 1
            break
    pca_component.update_pca(latent_space)

    # Free up memory before storing PCA
    pca_component.df = None
    pca_component.pcaStatistic = None

    # Store PCA object as file
    data_manager.store_dim_reduction_component(pca_component, "pca.pt")
    print(f'Trained PCA on latent space {latent_space}')

def isSquare(i):
    root = math.sqrt(i)
    return int(root + 0.5) ** 2 == i

def get_agent(strategy, env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy):
    if strategy.lower() in ["dqn"]:
        agent_class_name = DQNAgentLoop.__name__
        agent = DQNAgentLoop(env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy)
    else:
        agent_class_name = PPOAgentLoop.__name__
        agent = PPOAgentLoop(env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy)
    return agent_class_name, agent
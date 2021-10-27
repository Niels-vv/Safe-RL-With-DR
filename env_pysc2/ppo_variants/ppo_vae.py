import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
from env_pysc2.ppo_variants.ppo_base import AgentLoop as Agent
from utils.DataManager import DataManager
from vae.VAE import VAE, VaeManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name, load_policy):
        screen_size_x = env.observation_spec()[0].feature_screen[2]
        screen_size_y = env.observation_spec()[0].feature_screen[1]
        self.observation_space = screen_size_x * screen_size_y # iets met flatten van env.observation_spec() #TODO incorrect
            
        if not train_component: # We're not training vae, but using it in PPO
            vae_component = self.get_component(map_name)
            latent_space = vae_component.latent_space
            super(AgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name, load_policy, latent_space, vae_component)
        else:
            self.map = map_name
        self.train_component = train_component
        self.reduce_dim = not train_component
        self.vae = True

    def run_agent(self):
        if self.train_component:
            self.train_vae()
        else:
            super(AgentLoop, self).run_agent()

    def get_component(self, map):
        checkpoint = DataManager.get_network(f'env_pysc2/results_vae/{map}', "vae.pt")
        vae_model = VAE(in_channels = self.observation_space, latent_dim = checkpoint['latent_space']).to(device)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        vae_optimizer = optim.Adam(params=vae_model.parameters(), lr = checkpoint['vae_lr'])
        vae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        vae_model.eval()
        return VaeManager(vae_model, vae_optimizer, checkpoint['obs_file'],checkpoint['batch_size'], checkpoint['latent_space'], checkpoint['vae_lr'])

    def train_vae(self):
        # Hyperparameters VAE
        latent_space = 100 # TODO
        vae_lr = 0.0001 # TODO
        vae_batch_size = 10 # TODO

        # Create VAE model
        vae_model = VAE(in_channels = self.observation_space, latent_dim = latent_space).to(device)
        vae_optimizer = optim.Adam(params=vae_model.parameters(), lr=vae_lr)
        vae_manager = VaeManager(vae_model, vae_optimizer, f'env_pysc2/results_vae/{self.map}', vae_batch_size, latent_space, vae_lr)

        # Train VAE on observation trace
        self.data_manager = DataManager(observation_sub_dir = f'env_pysc2/observations/{self.map}', results_sub_dir = f'env_pysc2/results_vae/{self.map}')
        observation_trace = self.data_manager.get_observations()
        vae_manager.train_on_trace(observation_trace)

        # Store VAE
        checkpoint = vae_manager.get_checkpoint()
        self.data_manager.store_network(checkpoint, "vae.pt")


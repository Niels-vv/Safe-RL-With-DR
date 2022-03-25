import torch.optim as optim
import numpy as np
from dqn.dqn import Agent as DQNAgent
from utils.DataManager import DataManager
from autoencoder.ae import AE, AEManager

class AEAgent(DQNAgent):
    def __init__(self, env, dqn_config, device, max_episodes, data_manager, mlp, conv_last, encoder, deepmdp, train, train_ae_online, ae_encoder, ae_decoder, ae_config):
        ae_component = get_component(ae_config, ae_encoder, ae_decoder, train_ae_online, device)
        latent_space = ae_component.latent_space
        print(f'Latent space: {latent_space}')
        super(AEAgent, self).__init__(env, dqn_config, device, max_episodes, data_manager, mlp, conv_last, encoder, deepmdp, train)

        self.dim_reduction_component = ae_component
        self.reduce_dim = True
        self.ae = True
        self.train_ae_online = train_ae_online

    def run_agent(self):
        super(AEAgent, self).run_agent()

def get_component(ae_config, ae_encoder, ae_decoder, train_online, device):
    if train_online: # Not using pre-trained AE: initialize AE
        # Hyperparameters AE
        latent_space = ae_config['latent_space']
        ae_lr = ae_config['lr']
        ae_batch_size = ae_config['batch_size']

        ae_model = AE(ae_encoder, ae_decoder).to(device)
        ae_optimizer = optim.Adam(params=ae_model.parameters(), lr=ae_lr)
        ae_manager = AEManager(ae_model, ae_optimizer, ae_batch_size, latent_space, ae_lr)
        return ae_manager
        
    # Using pre-trained AE: load it
    checkpoint = DataManager.get_network(ae_config['file_path'], ae_config['file_name'], device)
    ae_model = AE(ae_encoder, ae_decoder).to(device)
    ae_model.load_state_dict(checkpoint['model_state_dict'])
    ae_optimizer = optim.Adam(params=ae_model.parameters(), lr = checkpoint['ae_lr'])
    ae_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ae_model.eval()
    ae_manager = AEManager(ae_model, ae_optimizer, checkpoint['batch_size'], checkpoint['latent_space'], checkpoint['ae_lr'])
    return ae_manager

def train_ae(ae_config, ae_encoder, ae_decoder, data_manager, device):
    # Hyperparameters AE
    latent_space = ae_config['latent_space']
    ae_lr = ae_config['lr']
    ae_batch_size = ae_config['batch_size']

    # Create AE model
    ae_model = AE(ae_encoder, ae_decoder).to(device)
    ae_optimizer = optim.Adam(params=ae_model.parameters(), lr=ae_lr)
    ae_manager = AEManager(ae_model, ae_optimizer, ae_batch_size, latent_space, ae_lr)

    # Train AE on observation traces
    i = 1
    while (data_manager.create_ae_results_dirs(i)): # Loop through all existing obs files
        print("Retreiving observations...")
        observation_trace = data_manager.get_observations()
        observation_trace = observation_trace.reshape(observation_trace.shape[0] * observation_trace.shape[1], 1, observation_trace.shape[2], observation_trace.shape[3]) # Reshape to 1 channel if there are multiple ones like with atari
        observation_trace = np.unique([tuple(obs) for obs in observation_trace],axis=0) # Remove duplicate obs
        print("Training AE...")
        ae_manager.train_on_trace(observation_trace)
        del observation_trace
        i += 1

    # Store AE
    print("Training done. Storing AE...")
    checkpoint = ae_manager.get_checkpoint()
    data_manager.store_network(checkpoint, ae_config['file_name'])

    
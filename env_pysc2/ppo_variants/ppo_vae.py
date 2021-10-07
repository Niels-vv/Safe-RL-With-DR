import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import pandas as pd
from env_pysc2.ppo_variants.ppo_base import AgentLoop as Agent
from utils.DataManager import DataManager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name):
        if not train_component:
            super(AgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name, self.get_latent_space())

        self.train_component = train_component
        self.reduce_dim = not train_component
        self.vae = True

    def run_agent(self):
        if self.train_component:
            self.train_vae()
        else:
            self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results_vae/{self.map}')
            self.dim_reduction_component  = self.data_manager.get_dim_reduction_component("vae.pt")
            super(AgentLoop, self).run_agent()

    def train_vae(self):
        # Hyperparameters VAE
        latent_space = 100 # TODO
        vae_lr = 0.0001 # TODO
        vae_batch_size = 10 # TODO

        # Create VAE model
        vae_model = VAE(in_channels = self.observation_space, latent_dim = latent_space).to(device)
        vae_optimizer = optim.Adam(params=vae_model.parameters(), lr=vae_lr)
        vae_manager = VaeManager(vae_model, vae_optimizer, vae_batch_size, latent_space)

        # Train VAE on observation trace
        self.data_manager = DataManager(observation_sub_dir = f'env_pysc2/observations/{self.map}', results_sub_dir = f'env_pysc2/results_vae/{self.map}')
        observation_trace = self.data_manager.get_observations()
        vae_manager.train_on_trace(observation_trace)

        # Store VAE
        self.data_manager.store_dim_reduction_component(vae_manager, "vae.pt")

    def get_latent_space(self):
        self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results_vae/{self.map}')
        self.dim_reduction_component = self.data_manager.get_dim_reduction_component("vae.pt")
        return self.dim_reduction_component.latent_space

class VAE(nn.Module):
# Use Linear instead of convs

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        out_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1],hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Linear(hidden_dims[-1],out_channels),
                            nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), input, mu, log_var

    def state_dim_reduction(self, state):
        mu, log_var = self.encode(state)
        z = self.reparameterize(mu, log_var)
        return z


    def loss_function(self, reconstruction, input, mu, log_var) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = reconstruction
        input = input
        mu = mu
        log_var = log_var

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}


class VaeManager():
  def __init__(self, vae_model, optimizer, obs_file, batch_size, latent_space):
    self.vae_model = vae_model
    self.optimizer = optimizer
    self.obs_file = obs_file
    self.batch_size = batch_size
    self.latent_space = latent_space

  def train_step(self, batch):
    reconstruction, input, mu, log_var = self.vae_model(batch)
    loss = self.vae_model.loss_function(reconstruction, input, mu, log_var)['loss']
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return loss

  def train_on_trace(self, obs):
    batch = []
    for index, row in obs.iterrows():
        batch.append(row)
        if len(batch) % self.batch_size == 0:
            self.train_step(torch.tensor(batch, dtype=torch.float, device=device))
            batch = []

  def state_dim_reduction(self, state):
    return self.vae_model.state_dim_reduction(state).squeeze()
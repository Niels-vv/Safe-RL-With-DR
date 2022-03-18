import torch, random
from torch import nn
from torch.nn import functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

class AE(nn.Module):
# Use Linear instead of convs

    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, input):
        result = self.encoder(input)
        return result

    def decode(self, z):
        result = self.decoder(z)
        return result

    def forward(self, input):
        z = self.encode(input)
        x = self.decode(z)
        return x

    def state_dim_reduction(self, state):
        state = state.unsqueeze(0)
        z = self.encode(state)
        return z


    def loss_function(self, reconstruction, input, mu, log_var):
        recons = reconstruction
        input = input
        mu = mu
        log_var = log_var

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kld_loss}


class AEManager():
    def __init__(self, ae_model, optimizer, batch_size, latent_space, ae_lr):
        self.ae_model = ae_model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.latent_space = latent_space
        self.ae_lr = ae_lr

    def train_step(self, batch):
        reconstruction = self.ae_model(batch)
        loss = F.mse_loss(batch, reconstruction, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f'Batch done. Loss: {loss}')
        return loss

    def train_on_trace(self, obs):
        obs = random.shuffle(obs)
        while len(obs) >= self.batch_size:
            batch = obs[-self.batch_size:]
            self.train_step(batch)
            del obs[-self.batch_size]

    def state_dim_reduction(self, state):
        return self.ae_model.state_dim_reduction(state).squeeze()

    def get_checkpoint(self):
        return {'model_state_dict' : self.ae_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'latent_space' : self.latent_space,
                'ae_lr' : self.ae_lr,
                'batch_size' : self.batch_size
        }
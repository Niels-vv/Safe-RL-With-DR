import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Based on https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

class VAE(nn.Module):
# Use Linear instead of convs

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()

        self.conv = True # Whether we are using input for CNN or linear
        self.latent_dim = latent_dim
        out_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(in_channels, h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim

        #self.encoder = nn.Sequential(*modules)
        c_hid = 32
        act_fn = nn.GELU
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
            act_fn(),
            nn.Flatten()
        )

        #self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_mu = nn.Linear(256*c_hid, latent_dim)
        self.fc_var = nn.Linear(256*c_hid, latent_dim)


        # Build Decoder
        modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        # hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.Linear(hidden_dims[i], hidden_dims[i+1]),
        #             nn.LeakyReLU())
        #     )



        #self.decoder = nn.Sequential(*modules)
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 256*c_hid),
            act_fn()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c_hid, 1, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
        )
        # self.final_layer = nn.Sequential(
        #                     nn.Linear(hidden_dims[-1],hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Linear(hidden_dims[-1],out_channels),
        #                     nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        print(f'Shape initial:{input.shape}')
        result = self.encoder(input)
        print(f'Shape after conv encode flatten {result.shape}')
        #result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        print(f'Shape after linear to latent {mu.shape}')
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        #result = self.decoder_input(z)
        ##result = result.view(-1, 512, 2, 2)
        #result = self.decoder(result)
        #result = self.final_layer(result)
        print(f'Shape before decode {z.shape}')
        x = self.linear(z)
        print(f'Shape after linear {x.shape}')
        x = x.reshape(x.shape[0], 1, 16, 16)
        print(f'Shape after reshape {x.shape}')
        result = self.decoder(x)
        print(f'Shape after decoding {result.shape}')
        raise NotImplementedError
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
    def __init__(self, vae_model, optimizer, obs_file, batch_size, latent_space, vae_lr):
        self.vae_model = vae_model
        self.optimizer = optimizer
        self.obs_file = obs_file
        self.batch_size = batch_size
        self.latent_space = latent_space
        self.vae_lr = vae_lr

    def train_step(self, batch):
        reconstruction, input, mu, log_var = self.vae_model(batch)
        loss = self.vae_model.loss_function(reconstruction, input, mu, log_var)['loss']
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(f'Batch done. Loss: {loss}')
        return loss

    def train_on_trace(self, obs):
        batch = []
        i = 0
        for index, row in obs.iterrows():
            i += 1
            row = row.to_numpy()
            if self.vae_model.conv:
                row = row.reshape(32,32) # TODO this is pysc2 specific
                row = np.expand_dims(row, 0)  
            batch.append(row)
            if len(batch) % self.batch_size == 0:
                batch = np.array(batch)
                #self.train_step(torch.tensor(batch, dtype=torch.float, device=device))
                self.train_step(torch.from_numpy(batch).to(device).float())
                batch = []
        print("donezo")

    def state_dim_reduction(self, state):
        return self.vae_model.state_dim_reduction(state).squeeze()

    def get_checkpoint(self):
        return {'model_state_dict' : self.vae_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'latent_space' : self.latent_space,
                'vae_lr' : self.vae_lr,
                'batch_size' : self.batch_size,
                'obs_file' : self.obs_file
        }

    def train_test_set(self,input):
        i = 0
        while i + self.batch_size < len(input):
            print(f'Traning images {i} - {i+self.batch_size}.')
            batch = input[i:i+self.batch_size]
            self.train_step(batch)
            i += self.batch_size

    def recons_test(self, input):
        for i in range(5):
            image = input[i:i+1]
            reconstruction, input, mu, log_var = self.vae_model(image)
            loss = self.vae_model.loss_function(reconstruction, input, mu, log_var)['loss']
            print(f'Loss: {loss}')
            fig = plt.figure
            plt.imshow(input[i], cmap='gray')
            plt.show()
            fig = plt.figure
            plt.imshow(reconstruction, cmap='gray')
            plt.show()

if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    vae_model = VAE(in_channels = 2, latent_dim = 16*16).to(device)
    vae_optimizer = optim.Adam(params=vae_model.parameters(), lr=0.001)
    vae_manager = VaeManager(vae_model, vae_optimizer, f'env_pysc2/results_vae/{map}', 100, 16*16, 0.001)
    vae_manager.train_test_set(mnist_trainset)
    vae_manager.recons_test(mnist_testset)
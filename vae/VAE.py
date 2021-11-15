import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np

import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

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
            nn.Conv2d(c_hid, 1, kernel_size=3, padding=1, stride=1),
            act_fn(),
            #nn.Flatten(), # Image grid to single feature vector
            #nn.Linear(256*c_hid, latent_dim)
        )

        #self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        #self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        #self.fc_mu = nn.Linear(256*c_hid, latent_dim)
        #self.fc_var = nn.Linear(256*c_hid, latent_dim)


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
        #self.linear = nn.Sequential(
        #    nn.Linear(latent_dim, 256*c_hid),
         #   act_fn()
        #)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1, stride=1),
            act_fn(),
            nn.Conv2d(c_hid, 1, kernel_size=3, padding=1, stride=1)
            
            #nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
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
        result = self.encoder(input)
        ##result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        #mu = self.fc_mu(result)
        #log_var = self.fc_var(result)
        return result

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
        #x = self.linear(z)
        #x = x.reshape(x.shape[0], -1, 16, 16)
        result = self.decoder(z)
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
        #mu, log_var = self.encode(input)
        #z = self.reparameterize(mu, log_var)
        #return  self.decode(z), input, mu, log_var
        z = self.encode(input)
        x = self.decode(z)
        return x

    def state_dim_reduction(self, state):
        #mu, log_var = self.encode(state)
        #z = self.reparameterize(mu, log_var)
        state = state.unsqueeze(0).unsqueeze(0)
        z = self.encode(state)
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
        #reconstruction, input, mu, log_var = self.vae_model(batch)
        #loss = self.vae_model.loss_function(reconstruction, input, mu, log_var)['loss']
        reconstruction = self.vae_model(batch)
        loss = F.mse_loss(batch, reconstruction, reduction="none")
        loss = loss.sum(dim=[1,2,3]).mean(dim=[0])
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
                print(f'Training observations {i-self.batch_size} - {i}')
                batch = np.array(batch)
                #self.train_step(torch.tensor(batch, dtype=torch.float, device=device))
                self.train_step(torch.from_numpy(batch).to(device).float())
                batch = []

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

    def train_test_set(self,dataloader):
        i=0
        for batch, _ in dataloader:
            print(f'Traning images {i} - {i+self.batch_size}.')
            self.train_step(batch.to(device).float())
            i+=self.batch_size

    def recons_test(self, batch):
        for i in range(5):
            image = batch[i].unsqueeze(0).to(device).float()
            reconstruction = self.vae_model(image)
            fig = plt.figure
            plt.imshow(batch[i].reshape(32,32), cmap='gray')
            plt.show()
            fig = plt.figure
            plt.imshow(reconstruction.reshape(32,32).cpu().detach(), cmap='gray')
            plt.show()

if __name__ == "__main__":
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose(
        [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
    ))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose(
        [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()]
    ))
    dataloader_train = torch.utils.data.DataLoader(mnist_trainset, 25, True)
    dataloader_test = torch.utils.data.DataLoader(mnist_testset, 5, True)
   
    vae_model = VAE(in_channels = 1, latent_dim = 16*16).to(device)
    vae_optimizer = optim.Adam(params=vae_model.parameters(), lr=0.0001)
    vae_manager = VaeManager(vae_model, vae_optimizer, f'env_pysc2/results_vae/{map}', 25, 16*16, 0.001)
    vae_manager.train_test_set(dataloader_train)
    vae_manager.recons_test(next(iter(dataloader_test))[0])
import torch
from torch import nn
from env_wrapper_abstract.env_wrapper import EnvWrapperAbstract
import numpy as np
from math import sqrt
from scipy.ndimage.interpolation import zoom
from collections import deque
import matplotlib.pyplot as plt
from torchvision import transforms

class SimpleCrop(nn.Module):
    """
    Crops an image (deterministically) using the transforms.functional.crop function. (No simple crop can be found in
    the torchvision.transforms library
    """

    def __init__(self, top: int, left: int, height: int, width: int) -> None:
        """
        See transforms.functional.crop for parameter descriptions
        """
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):
        """
        Forward pass for input img
        :param img: image tensor
        """
        return transforms.functional.crop(img, self.top, self.left, self.height, self.width)

class EnvWrapper(EnvWrapperAbstract):
    def __init__(self, env, device):
        self.env = env
        self.device = device
        self.ae_batch = []

        self.state_width = 84
        self.state_height = 84
        self.history_length = 4
        self.observation_stack = deque(maxlen = self.history_length) # 4 Stacked observations together make up one agent observation

        self.reward = 0
        self.episode = 0
        self.step_num = 0
        self.total_steps = 0
        self.duration = 0
        self.done = False

    def get_action(self, state, model, epsilon, train):
        # greedy
        if np.random.rand() > epsilon.value():
            state = torch.from_numpy(state).to(self.device).unsqueeze(0).float()
            with torch.no_grad():
                action = model(state).detach().cpu().data.numpy().squeeze()
            action = np.argmax(action)
        # explore
        else:
            action = self.env.action_space.sample()
        if train:
            epsilon.increment()
        return action

    def get_state(self, state, reduce_dim, reduction_component, pca, ae, latent_space):                 
        if reduce_dim:
            self.add_obs_to_ae_batch(state)
            self.reduce_dim(state, reduction_component, pca, ae, latent_space)           
        return state

    def step(self, action):
        self.total_steps += 1
        self.step_num += 1
        obs, reward, done, info = self.env.step(action)
        self.done = done
        self.observation_stack.append(obs)
        state = self.preprocess_screen()
        return state, reward, done

    def reset(self):
        self.done = False
        self.step_num = 0
        self.reward = 0
        self.episode += 1
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        for _ in range(self.history_length):
            self.observation_stack.append(obs)
        return self.preprocess_screen()

    def close(self):
        self.env.close()

    def is_last_obs(self):
        return self.done

    def reduce_dim(self, state, reduction_component, pca, ae, latent_space):
        state = torch.tensor(state, dtype=torch.float, device=self.device)
        state = reduction_component.state_dim_reduction(state)
        if ae: return state.detach().cpu().numpy() #Voor AE
        if pca: return torch.reshape(state, (int(sqrt(latent_space)), int(sqrt(latent_space)))).detach().cpu().numpy() #Voor Pca

    def add_obs_to_ae_batch(self, state):
        self.ae_batch.append(np.array(np.expand_dims(state, 0)))

    def preprocess_screen(self):
        trans = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((110, 84)), SimpleCrop(18, 0, 84, 84)]
         )
        obs_maxed_seq_arr = np.array(self.observation_stack)

        result = torch.tensor(obs_maxed_seq_arr)
        orig_device = result.device
        result = result.to(self.device)
        result = result.permute(0, 3, 1, 2)  # ensure "channels" dimension is in correct index
        result = trans(result)
        result = result.squeeze(1)           # Squeeze out grayscale dimension (original RGB dim)
        result = result.to(orig_device)  
        return np.array(result)

    def get_state_from_stored_obs(self, obs):
        return obs.reshape(self.history_length, self.state_height, self.state_width)

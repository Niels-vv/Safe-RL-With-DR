import torch
from env_wrapper.env_wrapper import EnvWrapperAbstract
import numpy as np
from math import sqrt
from scipy.ndimage.interpolation import zoom
from collections import deque

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
        self.step = 0
        self.total_steps = 0
        self.duration = 0
        self.done = False

    def get_action(self, state, model, epsilon, train):
        # greedy
        if np.random.rand() > self.epsilon.value():
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
        obs, reward, done, info = self.env.step(action)
        self.done = done
        self.observation_stack.append(obs)
        state = self.preprocess_screen()
        return state, reward, done


    def reset(self):
        self.done = False
        self.step = 0
        self.reward = 0
        self.episode += 1
        ob = self.env.reset()
        for _ in range(self.history_length):
            self.observation_stack.append(ob)
        return self.preprocess_screen()

    def close(self):
        pass

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
        if self.observation_stack[-1] is None:
            state = None
        else:
            gray = np.array(self.observation_stack).mean(3)
            gray_width = float(gray.shape[1])
            gray_height = float(gray.shape[2])
            state = zoom(gray,[1, self.state_width/gray_width, self.state_height/gray_height]).astype('float32')
        return state
import torch
from torch import nn
from pysc2.lib import actions, features
from env_wrapper_abstract.env_wrapper import EnvWrapperAbstract
import numpy as np
from math import sqrt

FUNCTIONS = actions.FUNCTIONS
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals

class EnvWrapper(EnvWrapperAbstract):
    def __init__(self, env, device, scripted_agent = False):
        self.env = env
        self.device = device
        self.ae_batch_len = 1000 # Gather 1000 frames for ae training
        self.ae_batch = np.empty((self.ae_batch_len, 1, 32, 32),dtype=np.float32)
        self.ae_index = 0

        self.screen_size = 32

        self.scripted = scripted_agent
        
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
            target = np.random.randint(0, self.screen_size, size=2)
            action = target[0] * self.screen_size + target[1]
        if train:
            epsilon.increment()
        return action

    def get_state(self, state, reduce_dim, reduction_component, pca, ae, latent_space, train_online):     
        state = state.observation.feature_screen.player_relative              
        if reduce_dim:
            if train_online:
                self.add_obs_to_ae_batch(state) # add last 84 x 84 frame to ae training batch
                if self.ae_index >= self.ae_batch_len: # Train AE on gathered frames
                    reduction_component.train_on_trace(self.ae_batch)
                    self.ae_batch = np.empty((self.ae_batch_len, 1, self.screen_size, self.screen_size),dtype=np.float32)
                    self.ae_index = 0

            state = self.reduce_dim(state, reduction_component, pca, ae, latent_space) # Reduce dim of last frame

        return state    # Return 1 x 32 x 32 or 1 x 16 x 16 state

    def step(self, action):
        self.total_steps += 1
        self.step_num += 1
        obs, reward, done, _ = self.env.step(action)
        self.done = done
        return obs, reward, done

    def reset(self):
        self.done = False
        self.resetted = True
        self.step_num = 0
        self.reward = 0
        self.episode += 1
        self.env.reset()
        return self.env.step([self.select_friendly_action()])[0] # Return state after selecting army unit

    def close(self):
        self.env.close()

    def is_last_obs(self):
        return self.done

    def reduce_dim(self, state, reduction_component, pca, ae, latent_space):
        if ae:
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0).unsqueeze(0) # Reshape to 1 X 1 X 84 X 84
            state = reduction_component.state_dim_reduction(state) 
            return state.detach().cpu().numpy()
        if pca: 
            state = reduction_component.state_dim_reduction(state) 
            return np.reshape(state, (1, int(sqrt(latent_space)), int(sqrt(latent_space))))

    def add_obs_to_ae_batch(self, state):
        if self.ae_index < self.ae_batch.shape[0]: 
            self.ae_batch[self.ae_index] = np.expand_dims(state, axis=0)
            self.ae_index += 1

    def get_loss(s,a,s_1,r, policy_network, target_network, gamma, multi_step):
        s_q  = policy_network(s)
        s_1_q = policy_network(s_1)
        s_1_target_q = target_network(s_1)

        selected_q = s_q.gather(1, a).squeeze(-1)
        s_1_q_max = s_1_q.max(1)[1]
        s_1_q_max.detach()

        s_1_target = s_1_target_q.gather(1, s_1_q_max[:,None]).squeeze(-1).detach()

        expected_q = r + (gamma ** multi_step) * s_1_target 

        loss = nn.MSELoss(selected_q, expected_q)
        return loss

    def select_friendly_action(self):
        return FUNCTIONS.select_army("select")

    def get_env_action(self, action, obs):
        def _xy_locs(mask):
            """Mask should be a set of bools from comparison with a feature layer."""
            y, x = mask.nonzero()
            return list(zip(x, y))
        
        if self.scripted:
            player_relative = obs.observation.feature_screen.player_relative
            beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            beacon_center = np.mean(beacon, axis=0).round()
            return FUNCTIONS.Move_screen("now", beacon_center)

        action = np.unravel_index(action, [self.screen_size, self.screen_size])
        target = [action[1], action[0]]
        
        command = _MOVE_SCREEN
        if command in obs.observation.available_actions:
            return actions.FunctionCall(command, [[0],target])
        else:
            return actions.FunctionCall(_NO_OP, [])
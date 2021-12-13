import torch, copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import abc
import numpy as np
from collections import deque
from itertools import chain
from utils.Epsilon import Epsilon
from utils.ReplayMemory import ReplayMemory
from deepmdp.DeepMDP import TransitionAux
from deepmdp.DeepMDP import compute_deepmdp_loss


# Based on https://github.com/alanxzhou/sc2bot/blob/master/sc2bot/agents/rl_agent.py
# DeepMDP based on paper en https://github.com/MkuuWaUjinga/DeepMDP-SSL4RL

seed = 0
torch.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

class MlpPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, deepmdp = False):
        super(MlpPolicy, self).__init__()
        self.deepmdp = deepmdp
        latent_dim = 256
        c_hid = 32
        act_fn = nn.GELU
        #self.linear = nn.Sequential(
        #    nn.Linear(latent_dim, 256*c_hid),
        #    act_fn()
        #)

        if deepmdp:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                act_fn(),
                nn.Conv2d(c_hid, 1, kernel_size=3, padding=1, stride=1),
                act_fn()
            )

        self.mlp = nn.Sequential(
            nn.ConvTranspose2d(1, c_hid, kernel_size=3, stride=2, padding=1, output_padding = 1), # 16 x 16 => 32 x 32
            #self.conv0 = nn.ConvTranspose2d(1, c_hid, kernel_size=3, stride=1, padding=1), # 32 x 32 => 32 x 32
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 1, kernel_size=3, stride=1, padding=1)
        )
        
        self.name = 'BeaconCNN'
        self.conv = True # Whether we're using a conv or linear output layer. Needed in def train_q(self)

    def forward(self, z, return_deepmdp = False):
        #x = self.linear(x)
        #x = x.reshape(x.shape[0], -1, 16, 16)
        if self.deepmdp:
            z = self.encoder(z)
        x = self.mlp(z)
        if return_deepmdp:
            return z, x
        else:
            return x

class AgentConfig:
    def __init__(self):
        self.config = {
                        # Learning
                        'train_q_per_step' : 4,
                        'gamma' : 0.99,
                        'train_q_batch_size' : 256,
                        'batches_before_training' : 20,
                        'target_q_update_frequency' : 250,
                        'lr' : 0.0001,
                        'plot_every' : 10,
                        'decay_steps' : 100000
        }

class Agent(AgentConfig):
    __metaclass__ = abc.ABCMeta

    def __init__(self, env, observation_space, action_space, max_steps, max_episodes):
        super(Agent, self).__init__()
        self.env = env
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.policy_network = MlpPolicy(self.latent_space, self.observation_space).to(self.device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr = self.config['lr'])
        self.epsilon = Epsilon(start=1.0, end=0.1, decay_steps=self.config['decay_steps'])
        self.loss = deque(maxlen=int(1e5))
        self.max_q = deque(maxlen=int(1e5))
        self.loss_history = []
        self.max_q_history = []
        self.reward_evaluation = []
        self.criterion = nn.MSELoss()
        self.max_gradient_norm = float('inf')
        self.memory = ReplayMemory(50000)

        # DeepMDP
        self.auxiliary_objective = None # Only used for DeepMDP (initialized in def setup_deepmdp(self))
        self.params = None
        self.penalty = 0.01

        self.data_manager = None
        self.deepmdp = False

    @abc.abstractmethod
    def run_loop(self):
        return

    @abc.abstractmethod
    def run_agent(self):
        return

    def setup_deepmdp(self):
        self.deepmdp = True
        self.policy_network = MlpPolicy(self.latent_space, self.observation_space, deepmdp = True).to(self.device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.auxiliary_objective = TransitionAux(self.device)
        self.params = [self.policy_network.parameters()] + [self.auxiliary_objective.network.parameters()]
        self.optimizer = optim.RMSprop(chain(*self.params), lr = self.config['lr'])

    def load_policy_checkpoint(self, checkpoint):
        self.policy_network.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        if checkpoint['deepmdp_state_dict'] is not None:
            self.auxiliary_objective = TransitionAux(self.device)
            self.auxiliary_objective.network.load_state_dict(checkpoint['deepmdp_state_dict'])
            self.params = [self.policy_network.parameters()] + [self.auxiliary_objective.network.parameters()]
            self.optimizer = optim.RMSprop(chain(*self.params), lr = self.config['lr'])
            self.auxiliary_objective.network.train()
        else:
            self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=checkpoint['lr'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon._value = checkpoint['epsilon']
        self.episode = checkpoint['episode']      
        if self.train:
            self.policy_network.train()
            self.target_network.train()
        else:
            self.policy_network.eval()
            self.target_network.eval()

    def get_policy_checkpoint(self):
        deepmdp_state_dict = None if self.auxiliary_objective is None else self.auxiliary_objective.network.state_dict()
        return {'policy_model_state_dict' : self.policy_network.state_dict(),
                'target_model_state_dict' : self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'deepmdp_state_dict' : deepmdp_state_dict,
                'lr' : self.config['lr'],
                'episode' : self.episode,
                'epsilon' : self.epsilon._value,
                'epsilon_t' : self.epsilon.t
        }

    def train_q(self):
        if self.config['train_q_batch_size'] >= len(self.memory):
            return

        s, a, s_1, r, done = self.memory.sample(self.config['train_q_batch_size'])
        s = torch.from_numpy(s).to(self.device).float()
        a = torch.from_numpy(a).to(self.device).long()
        s_1 = torch.from_numpy(s_1).to(self.device).float()
        r = torch.from_numpy(r).to(self.device).float()
        done = torch.from_numpy(done).to(self.device).float()

        if self.policy_network.conv:
            Q = self.policy_network(s, return_deepmdp = self.deepmdp)
            Qt = self.target_network(s_1).view(self.config['train_q_batch_size'], -1).detach()
            best_action = self.policy_network(s_1).view(self.config['train_q_batch_size'], -1).max(1)[1]
            if self.deepmdp:
                state_embeds = Q[0]
                Q = Q[1]
            Q = Q.view(self.config['train_q_batch_size'], -1)
        else:
            Q = self.policy_network(s) #TODO Not done for deepmdp; remove linear stuff if not used
            Qt = self.target_network(s_1).detach()
            best_action = self.policy_network(s_1).max(1)[1]
        Q = Q.gather(1, a)

        # double Q
        max_a_q_sp = Qt[
            np.arange(self.config['train_q_batch_size']), best_action].unsqueeze(1)
        y = r + (self.config['gamma'] * max_a_q_sp * (1-done))
        
        td_error = Q - y
        loss = td_error.pow(2).mul(0.5).mean()

        if self.deepmdp:
            new_states, _, _, _, _ = self.memory.sample(self.config['train_q_batch_size'])
            new_states  = torch.from_numpy(new_states).to(self.device).float()
            #print(f'Loss before deepmdp {loss}')
            loss += compute_deepmdp_loss(self.policy_network, self.auxiliary_objective, s, s_1, a, state_embeds, new_states, self.penalty, self.device)
            #print(f'Loss after deepmdp {loss}')

        self.loss.append(loss.sum().cpu().data.numpy())
        self.max_q.append(Q.max().cpu().data.numpy().reshape(-1)[0])
        self.optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        parameters = chain(*self.params) if self.deepmdp else self.policy_network.parameters()
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient_norm)
        self.optimizer.step()

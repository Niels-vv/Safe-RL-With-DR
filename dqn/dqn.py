import torch, copy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import abc
from utils.Epsilon import Epsilon
from utils.ReplayMemory import ReplayMemory
from collections import deque

# Based on https://github.com/alanxzhou/sc2bot/blob/master/sc2bot/agents/rl_agent.py

seed = 0
torch.manual_seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

class MlpPolicy(nn.Module):
    def __init__(self):
        super(MlpPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)
        self.name = 'BeaconCNN'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class AgentConfig:
    def __init__(self):
        self.config = {
                        # Learning
                        'train_q_per_step' : 4,
                        'gamma' : 0.99,
                        'train_q_batch_size' : 256,
                        'steps_before_training' : 10000,
                        'target_q_update_frequency' : 10000,
                        'lr' : 1e-8,
                        'plot_every' : 10
        }

class Agent(AgentConfig):
    __metaclass__ = abc.ABCMeta

    def __init__(self, env, observation_space, action_space, max_steps, max_episodes):
        super(Agent, self).__init__()
        self.env = env
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.policy_network = MlpPolicy().to(self.device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config['lr'])
        self.epsilon = Epsilon(start=0.9, end=0.1, update_increment=0.0001)
        self.loss = deque(maxlen=int(1e5))
        self.max_q = deque(maxlen=int(1e5))
        self.loss_history = []
        self.max_q_history = []
        self.criterion = nn.MSELoss()
        self.memory = ReplayMemory(50000)

        self.data_manager = None

    @abc.abstractmethod
    def run_loop(self):
        return

    @abc.abstractmethod
    def run_agent(self):
        return

    def load_policy_checkpoint(self, checkpoint):
        self.policy_network.load_state_dict(checkpoint['policy_model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=checkpoint['lr'])
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
        return {'policy_model_state_dict' : self.policy_network.state_dict(),
                'target_model_state_dict' : self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr' : self.config['lr'],
                'episode' : self.episode,
                'epsilon' : self.epsilon._value
        }

    def train_q(self, squeeze=False):
        if self.config['train_q_batch_size'] >= len(self.memory):
            return

        s, a, s_1, r, done = self.memory.sample(self.config['train_q_batch_size'])
        s = torch.from_numpy(s).to(self.device).float()
        a = torch.from_numpy(a).to(self.device).long().unsqueeze(1)
        s_1 = torch.from_numpy(s_1).to(self.device).float()
        r = torch.from_numpy(r).to(self.device).float()
        done = torch.from_numpy(1 - done).to(self.device).float()

        if squeeze:
            s = s.squeeze()
            s_1 = s_1.squeeze()

        # Q_sa = r + gamma * max(Q_s'a')
        Q = self.policy_network(s).view(self.config['train_q_batch_size'], -1)
        Q = Q.gather(1, a)

        Qt = self.target_network(s_1).view(self.config['train_q_batch_size'], -1)

        # double Q
        best_action = self.policy_network(s_1).view(self.config['train_q_batch_size'], -1).max(dim=1, keepdim=True)[1]
        y = r + done * self.config['gamma'] * Qt.gather(1, best_action)
        # y = r + done * self.gamma * Qt.max(dim=1)[0].unsqueeze(1)

        # y.volatile = False
        # with y.no_grad():
        loss = self.criterion(Q, y)
        self.loss.append(loss.sum().cpu().data.numpy())
        self.max_q.append(Q.max().cpu().data.numpy().reshape(-1)[0])
        self.optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        self.optimizer.step()
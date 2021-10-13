import torch
import torch.nn as nn
import torch.optim as optim
import abc

# Based on https://github.com/RPC2/PPO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MlpPolicy(nn.Module):
    def __init__(self, action_size, input_size=4):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.hidden_dims = [input_size*2, int(input_size/2),int(input_size/10),24]
        #self.hidden_dims = [512, 128, 24]
        # Build network
        modules = []
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_size, h_dim),
                    nn.ReLU())
            )
            input_size = h_dim

        self.fc = nn.Sequential(*modules)

        #self.fc1 = nn.Linear(self.input_size, 24)
        #self.fc2 = nn.Linear(24, 24)
        self.fc3_pi = nn.Linear(24, self.action_size)
        self.fc3_v = nn.Linear(24, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, x):
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc(x)
        x = self.fc3_pi(x)
        return self.softmax(x)

    def v(self, x):
        #x = self.relu(self.fc1(x))
        #x = self.relu(self.fc2(x))
        x = self.fc(x)
        x = self.fc3_v(x)
        return x

class AgentConfig:
    def __init__(self):
        self.config = {
                        # Learning
                        'gamma' : 0.99,
                        'plot_every' : 10,
                        'update_freq' : 10,
                        'k_epoch' : 3,
                        'learning_rate' : 0.0001,
                        'lmbda' : 0.95,
                        'eps_clip' : 0.2,
                        'v_coef' : 1,
                        'entropy_coef' : 0.01,

                        # Memory
                        'memory_size' : 400
        }

class Agent(AgentConfig):
    __metaclass__ = abc.ABCMeta

    def __init__(self, env, observation_space, action_space, max_steps, max_episodes):
        super(Agent, self).__init__()
        self.env = env
        self.max_steps = max_steps
        self.max_episodes = max_episodes
        self.policy_network = MlpPolicy(action_size=action_space, input_size = observation_space).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config['learning_rate'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['k_epoch'],
                                                   gamma=0.999) # TODO magic nr, same in load_policy_checkpoint
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.tensor([],dtype=torch.float)
        }

        self.data_manager = None

    @abc.abstractmethod
    def run_loop(self):
        return

    @abc.abstractmethod
    def run_agent(self):
        return

    def load_policy_checkpoint(self, checkpoint):
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=checkpoint['lr'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=checkpoint['step_size'],
                                                   gamma=checkpoint['gamma'])        
        if self.train:
            self.policy_network.train()
        else:
            self.policy_network.eval()

    def get_policy_checkpoint(self):
        return {'model_state_dict' : self.policy_network.state_dict(),
                'model_layers' : self.policy_network.hidden_dims,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr' : self.config['learning_rate'],
                'step_size' : self.config['k_epoch'],
                'gamma' : 0.999
        }

    def update_network(self):
        # get ratio
        pi = self.policy_network.pi(torch.tensor(self.memory['state'], dtype=torch.float, device=device))
        new_probs_a = torch.gather(pi, 1, torch.tensor(self.memory['action'], device= device))
        old_probs_a = torch.tensor(self.memory['action_prob'], dtype=torch.float, device=device)
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * torch.tensor(self.memory['advantage'], dtype=torch.float, device=device)
        surr2 = torch.clamp(ratio, 1 - self.config['eps_clip'], 1 + self.config['eps_clip']) * torch.tensor(self.memory['advantage'], dtype=torch.float, device=device)
        pred_v = self.policy_network.v(torch.tensor(self.memory['state'], dtype=torch.float, device=device))
        v_loss = (0.5 * (pred_v - self.memory['td_target']).pow(2)).to('cpu')  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = ((-torch.min(surr1, surr2)).to('cpu') + self.config['v_coef'] * v_loss - self.config['entropy_coef'] * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
 

    def add_memory(self, s, a, r, next_s, t, prob):
        if self.memory['count'] < self.config['memory_size']:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['action_prob'] = self.memory['action_prob'][1:]
            self.memory['advantage'] = self.memory['advantage'][1:]
            self.memory['td_target'] = self.memory['td_target'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append(prob)

    def finish_path(self, length):
        state = self.memory['state'][-length:]
        reward = self.memory['reward'][-length:]
        next_state = self.memory['next_state'][-length:]
        terminal = self.memory['terminal'][-length:]

        td_target = torch.tensor(reward, device=device) + \
                    self.config['gamma'] * self.policy_network.v(torch.tensor(next_state, dtype=torch.float, device=device)) * torch.tensor(terminal, device = device)
        delta = (td_target - self.policy_network.v(torch.tensor(state, dtype=torch.float, device=device))).to('cpu')
        delta = delta.detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.config['gamma'] * self.config['lmbda'] * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'].to(device), td_target.data), dim=0)
        self.memory['advantage'] += advantages
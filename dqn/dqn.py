import torch, time, copy, random, traceback, math, csv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import abc
import numpy as np
from collections import deque
from itertools import chain
from utils.Epsilon import Epsilon
from utils.ReplayMemory import ReplayMemory, Transition
from deepmdp.DeepMDP import TransitionAux
from deepmdp.DeepMDP import compute_deepmdp_loss

# Based on https://github.com/alanxzhou/sc2bot/blob/master/sc2bot/agents/rl_agent.py
# DeepMDP based on paper en https://github.com/MkuuWaUjinga/DeepMDP-SSL4RL

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


class MlpPolicy(nn.Module):
    def __init__(self, mlp, conv_last = True, encoder = None, deepmdp = False):
        super(MlpPolicy, self).__init__()
        self.deepmdp = deepmdp # Flag for whether we need to use the encoder
        self.encoder = encoder # Encoder for DeepMDP
        self.mlp = mlp         # Q* MLP
        self.conv = conv_last  # Whether we're using a conv or linear output layer. Needed in def train_q(self)

    def forward(self, z, return_deepmdp = False):
        if self.deepmdp:
            z = self.encoder(z)
        x = self.mlp(z)
        if return_deepmdp:
            return z, x
        else:
            return x


class Agent():
    __metaclass__ = abc.ABCMeta

    def __init__(self, env, config, device, max_episodes, data_manager, mlp, conv_last, encoder, deepmdp):

        self.env = env
        self.config = config
        self.device = device
        self.max_episodes = max_episodes
        self.duration = 0                   # Time duration of current episode
        self.data_manager = data_manager

        self.reduce_dim = False             # Whether to use dimensionality reduction on observations
        self.dim_reduction_component = None
        self.latent_space = None
        self.pca = False
        self.ae = False
        self.deepmdp = False
        self.train_ae_online = False        # When using an ae, whether it is pre-trained or not
        
        self.policy_network = MlpPolicy(mlp, conv_last, encoder, deepmdp).to(self.device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr = self.config['lr'])
        self.epsilon = Epsilon(start=1.0, end=0.1, decay_steps=self.config['decay_steps'])
        self.criterion = nn.MSELoss()
        self.max_gradient_norm = self.config['max_gradient_norm'] #float('inf')
        self.memory = ReplayMemory(50000)

        # DeepMDP (initialized in def setup_deepmdp(self))
        self.auxiliary_objective = None 
        self.params = None
        self.penalty = 0.01
        self.deepmdp = False

        self.reward_history = []
        self.duration_history = []
        self.epsilon_history = []

    def run_agent(self):
        print("Running dqn")
        # Setup file storage
        if self.train:
            self.data_manager.create_results_files()
        else:
            self.epsilon.isTraining = False # Act greedily

        # Run agent
        self.run_loop()

        # Store final results
        if self.train:
            variant = {'pca' : self.pca, 'ae' : self.ae, 'shield' : self.shield, 'latent_space' : self.latent_space}
            print("Rewards history:")
            for r in self.rewards:
                print(r)
            self.data_manager.write_results(self.reward_history, self.epsilon_history, self.duration_history, self.config, variant, self.get_policy_checkpoint())

    def reset(self):
        self.duration = 0
        return self.env.reset

    def run_loop(self):
        try:
            # A new episode
            while self.env.episode < self.max_episodes:
                state = self.reset()

                # Get state from env; applies dimensionality reduction if pca or ae are used
                state = self.env.get_state(state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space)
                
                # A step in an episode
                while True:
                    self.env.step += 1
                    self.env.total_steps += 1
                    start_duration = time.time()

                    # Choose action
                    action = self.env.get_action(state, self.policy_network, self.epsilon, self.train)

                    # Act
                    new_state, reward, done = self.env.step(action)

                    # Get new state observation; applies dimensionality reduction if pca or ae are used
                    new_state = self.env.get_state(new_state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space)

                    self.env.reward += reward

                    # Store transition to replay memory
                    if self.train: 
                        transition = Transition(state, action, new_state, reward, done)
                        self.memory.push(transition)

                    # Train Q network
                    if self.total_steps % self.config['train_q_per_step'] == 0 and self.total_steps > (self.config['batches_before_training'] * self.config['train_q_batch_size']) and self.epsilon.isTraining:
                        self.train_q()

                    # Update Target network
                    if self.total_steps % self.config['target_q_update_frequency'] == 0 and self.total_steps > (self.config['batches_before_training'] * self.config['train_q_batch_size']) and self.epsilon.isTraining:
                        for target, online in zip(self.target_network.parameters(), self.policy_network.parameters()):
                            target.data.copy_(online.data)
                    
                    state = new_state

                    # Train ae (if non-pretrained ae is used)
                    if self.train_ae_online and len(self.env.ae_batch) >= self.dim_reduction_component.batch_size: 
                        ae_batch = np.array(self.env.ae_batch)
                        self.dim_reduction_component.train_step(torch.from_numpy(ae_batch).to(self.device).float())
                        self.env.ae_batch = []

                    # Episode done
                    if self.env.is_last_obs():
                        # Train ae (if non-pretrained ae is used)
                        if self.train_ae_online and len(self.env.ae_batch) > 0: 
                            ae_batch = np.array(self.env.ae_batch)
                            self.dim_reduction_component.train_step(torch.from_numpy(ae_batch).to(self.device).float())
                            self.env.ae_batch = []

                        # Store and show info
                        end_duration = time.time()
                        self.duration += end_duration - start_duration

                        print(f'Episode {self.env.episode} done. Score: {self.env.reward}. Steps: {self.env.step}. Epsilon: {self.epsilon._value}')
                        self.reward_history.append(self.env.reward)
                        self.duration_history.append(self.env.duration)
                        self.epsilon_history.append(self.epsilon._value)

                        break

                # Store intermediate results in Google Drive
                if self.train and self.episode % self.config['intermediate_results_freq'] == 0: # TODO verbeteren in datamanager (kan gewoon results file in colab gebruiken, dus aangepaste write results aanroepen (want 'open "a"') oid)
                    eps = [x for x in range(len(self.reward_history))]
                    rows = zip(eps, self.reward_history, self.epsilon_history, self.duration_history)
                    try:
                        with open("/content/drive/MyDrive/Thesis/Code/PySC2/Results/Results.csv", "w") as f:
                            pass
                        with open("/content/drive/MyDrive/Thesis/Code/PySC2/Results/Results.csv", "a") as f:
                            writer = csv.writer(f)
                            writer.writerow(["Episode", "Reward", "Epsilon", "Duration"])
                            for row in rows:
                                writer.writerow(row)

                        torch.save(self.get_policy_checkpoint(), "/content/drive/MyDrive/Thesis/Code/PySC2/Results/policy_network.pt")
                    except Exception as e:
                        print("writing results failed")
                        print(e)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        finally:
            self.env.close()

    @abc.abstractmethod
    def run_agent(self):
        return

    def setup_deepmdp(self):
        self.deepmdp = True
        self.policy_network = MlpPolicy(self.latent_space, self.observation_space, deepmdp = True).to(self.device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.auxiliary_objective = TransitionAux(self.device)
        self.params = [self.policy_network.parameters()] + [self.auxiliary_objective.network.parameters()]
        self.optimizer = optim.Adam(chain(*self.params), lr = self.config['lr'])

    def load_policy_checkpoint(self, checkpoint):
        if checkpoint['deepmdp_state_dict'] is not None:
            self.setup_deepmdp()
            self.policy_network.load_state_dict(checkpoint['policy_model_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_model_state_dict'])

            self.auxiliary_objective.network.load_state_dict(checkpoint['deepmdp_state_dict'])

            self.params = [self.policy_network.parameters()] + [self.auxiliary_objective.network.parameters()]
            self.optimizer = optim.Adam(chain(*self.params), lr = self.config['lr'])
            if self.train: self.auxiliary_objective.network.train()
            print("Loaded DeepMDP DQN model")
        else:
            self.policy_network.load_state_dict(checkpoint['policy_model_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=checkpoint['lr'])
            print("Loaded DQN model")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon._value = checkpoint['epsilon']
        self.epsilon.t = checkpoint['epsilon_t']
        self.episode = checkpoint['episode'] - 1     
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

        self.optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        parameters = chain(*self.params) if self.deepmdp else self.policy_network.parameters()
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient_norm)
        self.optimizer.step()

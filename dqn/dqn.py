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
        self.mlp = mlp
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

    def __init__(self, env, config, device, max_episodes, data_manager, mlp, conv_last, encoder, deepmdp, train):

        self.env = env
        self.config = config
        self.device = device
        self.max_episodes = max_episodes
        self.duration = 0                   # Time duration of current episode
        self.data_manager = data_manager

        self.train = train 
        self.reduce_dim = False             # Whether to use dimensionality reduction on observations
        self.dim_reduction_component = None
        self.latent_space = None
        self.pca = False
        self.ae = False
        self.deepmdp = False
        self.train_ae_online = False        # When using an ae, whether it is pre-trained or not
        
        self.policy_network = MlpPolicy(mlp, conv_last, encoder, deepmdp).to(self.device)
        self.target_network = copy.deepcopy(self.policy_network)
        self.target_network.eval()
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr = self.config['lr'])
        self.epsilon = Epsilon(start=1.0, end=self.config['eps_end'], decay_steps=self.config['decay_steps'], train=self.train)
        self.criterion = nn.MSELoss()
        self.max_gradient_norm = self.config['max_gradient_norm'] #float('inf')
        self.memory = ReplayMemory(50000, min_train_buffer =  self.config['batches_before_training'] * self.config['train_q_batch_size'])

        # DeepMDP (initialized in def setup_deepmdp(self))
        self.auxiliary_objective = None 
        self.params = None
        self.penalty = 0.01
        self.deepmdp = False

        self.reward_history = []
        self.duration_history = []
        self.epsilon_history = []
        self.episode_history = []

    def run_agent(self):
        print("Running dqn")

        try:
            # Run agent
            self.run_loop()

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        finally:
            self.env.close()

            # Store final results
            if self.train:
                variant = {'pca' : self.pca, 'ae' : self.ae, 'latent_space' : self.latent_space}
                print("Rewards history:")
                for r in self.reward_history:
                    print(r)
                if self.ae and self.train_ae_online:
                    ae_checkpoint = self.dim_reduction_component.get_checkpoint()
                else:
                    ae_checkpoint = None
                self.data_manager.write_results(self.episode_history, self.reward_history, self.epsilon_history, self.duration_history, self.config, variant, self.get_policy_checkpoint(), ae_checkpoint)
                self.data_manager.write_intermediate_results(self.episode_history, self.reward_history, self.duration_history, self.epsilon_history, self.get_policy_checkpoint(), ae_checkpoint)

    def reset(self):
        self.duration = 0
        return self.env.reset()

    def run_loop(self):
        # A new episode
        while self.env.episode < self.max_episodes:
            state = self.reset()

            # Get state from env; applies dimensionality reduction if pca or ae are used
            state = self.env.get_state(state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space, self.train_ae_online)
            
            # A step in an episode
            while True:
                start_duration = time.time()

                # Choose action
                action = self.env.get_action(state, self.policy_network, self.epsilon, self.train)

                # Act
                new_state, reward, done = self.env.step(action)

                # Get new state observation; applies dimensionality reduction if pca or ae are used
                new_state = self.env.get_state(new_state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space, self.train_ae_online)

                self.env.reward += reward

                # Store transition to replay memory
                if self.train: 
                    transition = Transition(state, action, new_state, reward, done)
                    self.memory.push(transition)

                # Train Q network
                if self.epsilon.isTraining and self.env.total_steps % self.config['train_q_per_step'] == 0 and self.memory.ready_for_training():
                    self.train_q()

                # Update Target network
                if self.epsilon.isTraining and self.env.total_steps % self.config['target_q_update_frequency'] == 0 and self.memory.ready_for_training():
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                
                state = new_state

                # Train ae (if non-pretrained ae is used)
                # if self.train_ae_online and len(self.env.ae_batch) >= self.dim_reduction_component.batch_size: 
                #     ae_batch = np.array(self.env.ae_batch)
                #     self.dim_reduction_component.train_step(torch.from_numpy(ae_batch).to(self.device).float())
                #     self.env.ae_batch = []

                # Episode done
                if self.env.is_last_obs():
                    # Train ae (if non-pretrained ae is used)
                    # if self.train_ae_online and len(self.env.ae_batch) > 0: 
                    #     ae_batch = np.array(self.env.ae_batch)
                    #     self.dim_reduction_component.train_step(torch.from_numpy(ae_batch).to(self.device).float())
                    #     self.env.ae_batch = []

                    # Store and show info
                    end_duration = time.time()
                    self.duration = end_duration - start_duration

                    if self.env.episode % self.config['plot_every'] == 0:
                        print(f'Episode {self.env.episode} done. Score: {self.env.reward}. Steps: {self.env.step_num}. Epsilon: {self.epsilon._value}. Fps: {self.env.step_num / self.duration}')
                    self.reward_history.append(self.env.reward)
                    self.duration_history.append(self.env.duration)
                    self.epsilon_history.append(self.epsilon._value)
                    self.episode_history.append(self.env.episode)

                    break

            # Store intermediate results in Google Drive
            if self.train and self.env.episode % self.config['intermediate_results_freq'] == 0:
                print("Storing intermediate results")
                if self.ae and self.train_ae_online:
                    ae_checkpoint = self.dim_reduction_component.get_checkpoint()
                else:
                    ae_checkpoint = None
                self.data_manager.write_intermediate_results(self.episode_history, self.reward_history, self.duration_history, self.epsilon_history, self.get_policy_checkpoint(), ae_checkpoint)

    def fill_buffer(self):
        print("Filling replay memory buffer...")
        # Retain current settings
        current_eps = self.env.episode
        current_steps = self.env.total_steps

        episode = 0
        # A new episode
        while not self.memory.is_filled():
            state = self.reset()

            # Get state from env; applies dimensionality reduction if pca or ae are used
            state = self.env.get_state(state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space, train_online = False)
            
            # A step in an episode
            while True:
                # Choose action. Set train to False so we don't increment epsilon
                action = self.env.get_action(state, self.policy_network, self.epsilon, train = False)

                # Act
                new_state, reward, done = self.env.step(action)

                # Get new state observation; applies dimensionality reduction if pca or ae are used
                new_state = self.env.get_state(new_state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space, train_online = False)

                self.env.reward += reward

                # Store transition to replay memory
                if self.train: 
                    transition = Transition(state, action, new_state, reward, done)
                    self.memory.push(transition)
                
                state = new_state

                # Episode done
                if self.env.is_last_obs():
                    print(f'Filling memory buffer episode {episode} done.')
                    episode += 1
                    break

        self.env.episode = current_eps
        self.env.total_steps = current_steps
        print("Done filling Memory.")

    def store_observations(self, total_obs, observations, skip_frame):
        self.data_manager.create_observation_file()

        stored_obs = 0
        may_store = False
        
        print("Storing observations...")

        # A new episode
        while stored_obs < total_obs:
            state = self.reset()

            # Get state from env; applies dimensionality reduction if pca or ae are used
            state = self.env.get_state(state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space, self.train_ae_online)

            # A step in an episode
            while stored_obs < total_obs:
                
                # Choose action. Set train to False so we don't increment epsilon
                action = self.env.get_action(state, self.policy_network, self.epsilon, train = False)

                # Act
                new_state, reward, _ = self.env.step(action)
                self.env.reward += reward
                
                # Store observation
                if self.env.total_steps % skip_frame == 0:
                    observations[stored_obs] = new_state
                    stored_obs += 1
                    may_store = True
                if stored_obs % 1000 == 0 and may_store:
                    self.data_manager.store_observation(observations[:stored_obs])
                    may_store = False
                    print(f'Stored {stored_obs} / {total_obs} observations.')

                # Get new state observation;
                new_state = self.env.get_state(new_state, self.reduce_dim, self.dim_reduction_component, self.pca, self.ae, self.latent_space, self.train_ae_online)
                
                state = new_state

                # Episode done
                if self.env.is_last_obs():
                    print(f'Episode reward: {self.env.reward}')
                    break
        print("Storing final obs...")
        self.data_manager.store_observation(observations)
        print("Done")

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
        self.env.episode = checkpoint['episode'] - 1     
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
                'episode' : self.env.episode,
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

        loss = self.get_loss(s,a,s_1,r,done)

        self.optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        parameters = chain(*self.params) if self.deepmdp else self.policy_network.parameters()
        torch.nn.utils.clip_grad_norm_(parameters, self.max_gradient_norm)
        self.optimizer.step()

    def get_loss(self, s, a, s_1, r, done):
        if self.policy_network.conv:
            Q = self.policy_network(s, return_deepmdp = self.deepmdp)
            Qt = self.target_network(s_1).view(self.config['train_q_batch_size'], -1).detach()
            best_action = self.policy_network(s_1).view(self.config['train_q_batch_size'], -1).max(1)[1]
            if self.deepmdp:
                state_embeds = Q[0]
                Q = Q[1]
            Q = Q.view(self.config['train_q_batch_size'], -1)
        else:
            Q = self.policy_network(s) #TODO Not done for deepmdp;
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
        return loss

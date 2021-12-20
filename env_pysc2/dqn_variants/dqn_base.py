from matplotlib.pyplot import flag
import torch, time, copy, random, traceback, math, csv
import numpy as np
from dqn.dqn import Agent
from pysc2.lib import actions
from pysc2.lib import features
from utils.DataManager import DataManager
from utils.Epsilon import Epsilon
from utils.ReplayMemory import ReplayMemory, Transition
from math import sqrt

# For info on obs from env.step, see: https://github.com/deepmind/pysc2/blob/master/pysc2/env/environment.py and https://github.com/deepmind/pysc2/blob/master/docs/environment.md
# Based on https://github.com/alanxzhou/sc2bot/blob/master/sc2bot/agents/rl_agent.py

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

_PLAYER_FRIENDLY = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_HOSTILE = features.PlayerRelative.ENEMY
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_SELECT_POINT = actions.FUNCTIONS.select_point.id

FUNCTIONS = actions.FUNCTIONS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

''' Base class for DQN agent in pysc2.'''
class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, map_name, load_policy, latent_space = None, dim_reduction_component = None):
        self.device = device
        self.screen_size_x = 32#env.observation_spec()[0].feature_screen[2]
        self.screen_size_y = 32#env.observation_spec()[0].feature_screen[1]
        self.observation_space = self.screen_size_x * self.screen_size_y # iets met flatten van env.observation_spec() #TODO incorrect
        self.action_space = self.screen_size_x * self.screen_size_y # iets met flatten van env.action_spec()    #TODO incorrect
        self.latent_space = latent_space if latent_space is not None else self.observation_space
        self.train = train 
        
        super(AgentLoop, self).__init__(env, self.latent_space, self.action_space, max_steps, max_episodes)

        self.reduce_dim = False
        self.shield = shield
        self.pca = False
        self.vae = False
        self.deepmdp = False
        self.train_ae_online = False
        self.dim_reduction_component = dim_reduction_component
        self.map = map_name

        self.reward = 0
        self.episode = 0
        self.step = 0
        self.total_steps = 0
        self.duration = 0
        self.obs_spec = env.observation_spec()[0]
        self.action_spec = env.action_spec()[0]

        if load_policy:
            checkpoint = DataManager.get_network(f'env_pysc2/results/dqn/{map_name}', "policy_network.pt", self.device)
            self.load_policy_checkpoint(checkpoint)
            eps = self.episode
            if self.train:
                print("Running episodes to refill memory buffer.")
                with torch.no_grad():
                    self.fill_buffer(eps)
                self.episode = eps
                print("Running training episodes.")


    def get_env_action(self, action, obs):
        #player_relative = obs.observation.feature_screen.player_relative
        #beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
        #beacon_center = np.mean(beacon, axis=0).round()
        #print(beacon_center)
        #return FUNCTIONS.Move_screen("now", beacon_center)

        action = np.unravel_index(action, [self.screen_size_x, self.screen_size_y])
        target = [action[1], action[0]]
        
        command = _MOVE_SCREEN
        if command in obs.observation.available_actions:
            #return FUNCTIONS.move_screen("now",target)
            return actions.FunctionCall(command, [[0],target])
        else:
            return actions.FunctionCall(_NO_OP, [])

    # TODO: unsqueeze=False gedaan 
    def get_action(self, s, unsqueeze=True):
        # greedy
        if np.random.rand() > self.epsilon.value():
            s = torch.from_numpy(s).to(self.device)
            if unsqueeze:
                s = s.unsqueeze(0).float()
            else:
                s = s.float()
            with torch.no_grad():
                action = self.policy_network(s).detach().cpu().data.numpy().squeeze()
            action = np.argmax(action)
        # explore
        else:
            #action = 0
            target = np.random.randint(0, self.screen_size_x, size=2)
            #action =  action * self.screen_size_x * self.screen_size_y + target[0] * self.screen_size_x + target[1]
            action = target[0] * self.screen_size_x + target[1]
        if self.train:
            self.epsilon.increment()
        return action

    def select_friendly_action(self):
        return FUNCTIONS.select_army("select")
    
    def reset(self):
        self.step = 0
        self.reward = 0
        self.duration = 0
        self.episode += 1
        self.env.reset()
        select_friendly = self.select_friendly_action()
        return self.env.step([select_friendly])

    def run_agent(self):
        print("Running dqn")
        # Setup file storage
        if self.train:
            self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results/dqn/{self.map}')
            self.data_manager.create_results_files()
        else:
            self.epsilon.isTraining = False # Act greedily

        # Run agent
        rewards, epsilons, durations = self.run_loop()

        # Store results
        if self.train:
            variant = {'pca' : self.pca, 'vae' : self.vae, 'shield' : self.shield, 'latent_space' : self.latent_space}
            print("Rewards history:")
            for r in rewards:
                print(r)
            self.data_manager.write_results(rewards, epsilons, durations, self.config, variant, self.get_policy_checkpoint())


    def fill_buffer(self, eps):
        episode = 0
        while episode < (min(eps, int(math.ceil(self.memory.capacity / 239)))):
            obs = self.reset()[0]

            state = obs.observation.feature_screen.player_relative                    
            if self.reduce_dim:
                state = torch.tensor(state, dtype=torch.float, device=device)
                state = self.dim_reduction_component.state_dim_reduction(state)
                state = state.detach().cpu().numpy()
                #state = torch.reshape(state, (int(sqrt(self.latent_space)), int(sqrt(self.latent_space)))).detach().cpu().numpy()
            state = np.expand_dims(state, 0)
            
            # A step in an episode
            while self.step < self.max_steps:
                self.step += 1
                self.total_steps += 1
                # Choose action
                self.train = False
                action = self.get_action(state)
                self.train = True

                # Act
                act = self.get_env_action(action, obs)
                obs = self.env.step([act])[0]

                # Get state observation
                new_state = obs.observation.feature_screen.player_relative              
                if self.reduce_dim:
                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                    new_state = self.dim_reduction_component.state_dim_reduction(new_state)
                    new_state = new_state.detach().cpu().numpy()
                    #new_state = torch.reshape(new_state, (int(sqrt(self.latent_space)), int(sqrt(self.latent_space)))).detach().cpu().numpy()
                new_state = np.expand_dims(new_state, 0)   

                reward = obs.reward
                #reward = -1 if terminal else reward
                terminal = reward > 0 # Agent found beacon

                if self.train: 
                    transition = Transition(state, action, new_state, reward, terminal)
                    self.memory.push(transition)
                
                state = new_state

                # 120s passed, i.e. episode done
                if obs.last():
                    episode += 1
                    break



    def run_loop(self, evaluate_checkpoints = 15):
        reward_history = []
        duration_history = []
        step_history = []
        epsilon_history = []
        avg_reward = []

        try:
            # A new episode
            while self.episode < self.max_episodes:
                ae_batch = []
                obs = self.reset()[0]

                state = obs.observation.feature_screen.player_relative                    
                if self.reduce_dim:
                    if self.train_ae_online: ae_batch.append(np.array(np.expand_dims(state, 0)))
                    state = torch.tensor(state, dtype=torch.float, device=device)
                    state = self.dim_reduction_component.state_dim_reduction(state)
                    state = state.detach().cpu().numpy()
                    #state = torch.reshape(state, (int(sqrt(self.latent_space)), int(sqrt(self.latent_space)))).detach().cpu().numpy()
                state = np.expand_dims(state, 0)
                
                # A step in an episode
                while self.step < self.max_steps:
                    self.step += 1
                    self.total_steps += 1
                    start_duration = time.time()
                    # Choose action
                    if self.train:
                        action = self.get_action(state)
                    else:
                        with torch.no_grad():
                            action = self.get_action(state)
                            # action = torch.distributions.Categorical(prob_a).sample().item()

                    # Act
                    act = self.get_env_action(action, obs)
                    obs = self.env.step([act])[0]

                    # Get state observation
                    new_state = obs.observation.feature_screen.player_relative              
                    if self.reduce_dim:
                        if self.train_ae_online: ae_batch.append(np.array(np.expand_dims(new_state, 0)))
                        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                        new_state = self.dim_reduction_component.state_dim_reduction(new_state)
                        new_state = new_state.detach().cpu().numpy()
                        #new_state = torch.reshape(new_state, (int(sqrt(self.latent_space)), int(sqrt(self.latent_space)))).detach().cpu().numpy()
                    new_state = np.expand_dims(new_state, 0)   

                    reward = obs.reward
                    #reward = -1 if terminal else reward
                    terminal = reward > 0 # Agent found beacon
                    self.reward += reward

                    if self.train: 
                        transition = Transition(state, action, new_state, reward, terminal)
                        self.memory.push(transition)

                        if self.episode % 15 == 0: # TODO verbeteren in datamanager (kan gewoon results file in colab gebruiken, dus aangepaste write results aanroepen (want 'open "a"') oid)
                            eps = [x for x in range(len(reward_history))]
                            rows = zip(eps, reward_history, epsilon_history, duration_history)
                            try:
                                with open("/content/../Results.csv", "w") as f:
                                    pass
                                with open("/content/../Results.csv", "a") as f:
                                    writer = csv.writer(f)
                                    writer.writerow(["Episode", "Reward", "Epsilon", "Duration"])
                                    for row in rows:
                                        writer.writerow(row)

                                torch.save(self.get_policy_checkpoint(), "/content/../policy_network.pt")
                            except Exception as e:
                                print("writing results failed")
                                print(e)

                    if self.total_steps % self.config['train_q_per_step'] == 0 and self.total_steps > (self.config['batches_before_training'] * self.config['train_q_batch_size']) and self.epsilon.isTraining:
                        self.train_q()

                    if self.total_steps % self.config['target_q_update_frequency'] == 0 and self.total_steps > (self.config['batches_before_training'] * self.config['train_q_batch_size']) and self.epsilon.isTraining:
                        for target, online in zip(self.target_network.parameters(), self.policy_network.parameters()):
                            target.data.copy_(online.data)
                    
                    state = new_state

                    if self.train_ae_online and len(ae_batch) >= self.dim_reduction_component.batch_size: 
                        ae_batch = np.array(ae_batch)
                        self.dim_reduction_component.train_step(torch.from_numpy(ae_batch).to(device).float())
                        ae_batch = []

                    # 120s passed, i.e. episode done
                    if obs.last():
                        if self.train_ae_online and len(ae_batch) > 0: 
                            ae_batch = np.array(ae_batch)
                            self.dim_reduction_component.train_step(torch.from_numpy(ae_batch).to(device).float())

                        end_duration = time.time()
                        self.duration += end_duration - start_duration

                        print(f'Episode {self.episode} done. Score: {self.reward}. Steps: {self.step}. Epsilon: {self.epsilon._value}')
                        reward_history.append(self.reward)
                        duration_history.append(self.duration)
                        epsilon_history.append(self.epsilon._value)
                        avg_reward.append(sum(reward_history[-10:])/10.0)

                        
                        #print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                        #   'loss: %.4f, lr: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss,
                        #                                self.scheduler.get_last_lr()[0]))

                        #self.env.reset()

                        break

                if evaluate_checkpoints > 0 and ((self.episode % evaluate_checkpoints) - (evaluate_checkpoints - 1) == 0 or self.episode == 0):
                    print('Evaluating...')
                    self.epsilon.isTraining = False  # we need to make sure that we act greedily when we evaluate
                    tr = self.train
                    self.train = False
                    with torch.no_grad():
                        self.run_loop(evaluate_checkpoints=0)
                    self.train = tr
                    self.epsilon.isTraining = True
                if evaluate_checkpoints == 0:  # this should only activate when we're inside the evaluation loop
                    #self.reward_evaluation.append(self.reward)
                    print(f'Evaluation Complete: Episode reward = {self.reward}')
                    break

                if len(self.loss) > 0:
                    self.loss_history.append(self.loss[-1])
                    self.max_q_history.append(self.max_q[-1])

                if self.episode % self.config['plot_every'] == 0:
                    pass #plot             
        except KeyboardInterrupt:
           pass
        except Exception as e:
           print(e)
           print(traceback.format_exc())
        finally:
           if evaluate_checkpoints > 0:
             self.env.close()
             return reward_history, epsilon_history, duration_history

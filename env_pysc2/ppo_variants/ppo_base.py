from matplotlib.pyplot import flag
import torch, time
import numpy as np
from ppo.PPO import Agent
from pysc2.lib import actions
from pysc2.lib import features
from utils.DataManager import DataManager

# Based on: https://github.com/whathelll/DeepRLBootCampLabs, https://github.com/whathelll/DeepRLBootCampLabs/blob/master/pytorch/sc2_agents/base_rl_agent.py
# For info on obs from env.step, see: https://github.com/deepmind/pysc2/blob/master/pysc2/env/environment.py and https://github.com/deepmind/pysc2/blob/master/docs/environment.md

seed = 3
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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

''' Base class for PPO in pysc2.'''
class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, store_observations, map_name, load_policy, latent_space = None):
        self.screen_size_x = env.observation_spec()[0].feature_screen[2]
        self.screen_size_y = env.observation_spec()[0].feature_screen[1]
        self.observation_space = self.screen_size_x * self.screen_size_y # iets met flatten van env.observation_spec() #TODO incorrect
        self.action_space = self.screen_size_x * self.screen_size_y # iets met flatten van env.action_spec()    #TODO incorrect
        self.latent_space = latent_space if latent_space is not None else self.observation_space
        super(AgentLoop, self).__init__(env, self.latent_space, self.action_space, max_steps, max_episodes)

        self.train = train 
        self.reduce_dim = False
        self.shield = shield
        self.pca = False
        self.vae = False
        self.dim_reduction_component = None
        self.store_obs = store_observations
        self.load_policy = load_policy
        self.map = map_name

        self.reward = 0
        self.episode = 0
        self.step = 0
        self.duration = 0
        self.obs_spec = env.observation_spec()[0]
        self.action_spec = env.action_spec()[0]

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
        # Setup file storage
        if self.store_obs:
            self.data_manager = DataManager(observation_sub_dir=f'env_pysc2/observations/{self.map}')
            self.data_manager.create_observation_file()
            self.train = False
        if self.train:
            self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results/{self.map}')
            self.data_manager.create_results_files()

        # TODO load corresponding optimizer and scheduler
        if self.load_policy:
            self.policy_network = DataManager.get_component(f'env_pysc2/results/{self.map}', "policy_network.pt")

        # Run agent
        rewards, steps, durations = self.run_loop()

        # Store results
        if self.train:
            variant = {'pca' : self.pca, 'vae' : self.vae, 'shield' : self.shield}
            self.data_manager.write_results(rewards, steps, durations, self.config, variant, self.policy_network)

    def run_loop(self):
        reward_history = []
        duration_history = []
        step_history = []
        avg_reward = []

        try:
            # A new episode
            while self.episode < self.max_episodes:
                print("new episode")
                obs = self.reset()[0]
                
                # Get initial state
                state = obs.observation.feature_screen.player_relative.flatten()
                state_mem = state
                state = torch.tensor(state, dtype=torch.float, device=device)
                if self.reduce_dim:
                    state = self.dim_reduction_component.state_dim_reduction(state)
                    state_mem = state.tolist()
                if self.store_obs: self.data_manager.store_observation(state_mem)

                # A step in an episode
                while self.step < self.max_steps:
                    self.step += 1

                    start_duration = time.time()
                    # Choose action
                    if self.train:
                        prob_a = self.policy_network.pi(state)
                        action = torch.distributions.Categorical(prob_a).sample().item()
                    else:
                        with torch.no_grad():
                            prob_a = self.policy_network.pi(state)
                            action = torch.distributions.Categorical(prob_a).sample().item()

                    # Act
                    act = self.get_env_action(action, obs)
                    end_duration = time.time()
                    self.duration += end_duration - start_duration
                    obs = self.env.step([act])[0]
                    new_state = obs.observation.feature_screen.player_relative.flatten()
                    new_state_mem = new_state
                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                    if self.reduce_dim:
                        new_state = self.dim_reduction_component.state_dim_reduction(new_state)
                        new_state_mem = new_state.tolist()

                    if self.store_obs: self.data_manager.store_observation(state_mem)

                    reward = obs.reward
                    terminal = reward > 0 # Agent found beacon
                    self.reward += reward

                    #reward = -1 if terminal else reward
                    if self.train: self.add_memory(state_mem, action, reward, new_state_mem, terminal, prob_a[action].item())

                    state = new_state
                    state_mem = new_state_mem

                    # 120s passed, i.e. episode done
                    if obs.last():
                        print(f'Episode {self.episode} done. Score: {self.reward}. Steps: {self.step}.')
                        reward_history.append(self.reward)
                        duration_history.append(self.duration)
                        step_history.append(self.step)
                        avg_reward.append(sum(reward_history[-10:])/10.0)

                        start_duration = time.time()
                        if self.train: self.finish_path(self.step)
                        end_duration = time.time()
                        self.duration += end_duration - start_duration

                        #print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                        #   'loss: %.4f, lr: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss,
                        #                                self.scheduler.get_last_lr()[0]))

                        #self.env.reset()

                        break
                        
                if self.train and self.episode % self.config['update_freq'] == 0:
                    for _ in range(self.config['k_epoch']):
                        self.update_network()

                if self.episode % self.config['plot_every'] == 0:
                    pass #plot

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            self.env.close()
            return reward_history, step_history, duration_history
import torch
import numpy as np
from ppo.PPO import Agent
from pysc2.lib import actions
from pysc2.lib import features
from torch.autograd import Variable

# Based on: https://github.com/whathelll/DeepRLBootCampLabs, https://github.com/whathelll/DeepRLBootCampLabs/blob/master/pytorch/sc2_agents/base_rl_agent.py
# For info on obs from env.step, see: https://github.com/deepmind/pysc2/blob/master/pysc2/env/environment.py and https://github.com/deepmind/pysc2/blob/master/docs/environment.md

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
class AgentLoop(Agent):
    def __init__(self, env, max_steps, max_episodes, test=False):
        self.screen_size_x = env.observation_spec()[0].feature_screen[2]
        self.screen_size_y = env.observation_spec()[0].feature_screen[1]
        observations_space = self.screen_size_x * self.screen_size_y # iets met flatten van env.observation_spec() #TODO incorrect
        action_space = self.screen_size_x * self.screen_size_y # iets met flatten van env.action_spec()    #TODO incorrect
        super(AgentLoop, self).__init__(env, observations_space, action_space, max_steps, max_episodes, test)

        self.reward = 0
        self.episode = 0
        self.step = 0
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
        self.episode += 1
        self.env.reset()
        select_friendly = self.select_friendly_action()
        return self.env.step([select_friendly])

    def run_loop(self):
        reward_history = []
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

                # A step in an episode
                while self.step < self.max_steps:
                    self.step += 1

                    # Choose action
                    prob_a = self.policy_network.pi(state)
                    action = torch.distributions.Categorical(prob_a).sample().item()
                    
                    # Act
                    act = self.get_env_action(action, obs)
                    obs = self.env.step([act])[0]
                    new_state = obs.observation.feature_screen.player_relative.flatten()
                    new_state_mem = new_state
                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                    
                    reward = obs.reward
                    terminal = obs.last()
                    self.reward += reward

                    #reward = -1 if terminal else reward

                    self.add_memory(state_mem, action, reward, new_state_mem, terminal, prob_a[action].item())

                    state = new_state
                    state_mem = new_state_mem

                    if terminal:
                        print("Terminal")
                        reward_history.append(self.reward)
                        avg_reward.append(sum(reward_history[-10:])/10.0)

                        self.finish_path(self.step)

                        #print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                        #   'loss: %.4f, lr: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss,
                        #                                self.scheduler.get_last_lr()[0]))

                        #self.env.reset()

                        break
                        
                if self.episode % self.update_freq == 0:
                    for _ in range(self.k_epoch):
                        print("updating network")
                        self.update_network()

                if self.episode % self.plot_every == 0:
                    pass #plot

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            self.env.close()
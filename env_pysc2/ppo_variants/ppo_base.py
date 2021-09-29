import torch
import numpy as np
from ppo.PPO import Agent
from pysc2.lib import actions
from pysc2.lib import features
from torch.autograd import Variable

# Based on: https://github.com/whathelll/DeepRLBootCampLabs, https://github.com/whathelll/DeepRLBootCampLabs/blob/master/pytorch/sc2_agents/base_rl_agent.py

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

class AgentLoop(Agent):
    def __init__(self, env, max_steps, max_episodes, test=False):
        observations_space = 10 # iets met flatten van env.observation_spec() #TODO incorrect
        action_space = 1 # iets met flatten van env.action_spec()    #TODO incorrect
        super(AgentLoop, self).__init__(env, observations_space, action_space, max_steps, max_episodes, test)

        self.reward = 0
        self.episode = 0
        self.step = 0
        self.screen_size = env.observation_spec()[0].feature_screen[1]
        self.obs_spec = None
        self.action_spec = None

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def get_env_action(self, action, obs):
        action = np.unravel_index(action, [self._screen_size, self._screen_size])
        target = [action[1], action[0]]
        command = _MOVE_SCREEN 
        if command in obs.observation.available_actions:
            return actions.FunctionCall(command, [[0], target])
        else:
            return actions.FunctionCall(_NO_OP, [])


    '''
        :param
        s = obs.observation["screen"]
        :returns
        action = argmax action
    '''
    def get_action(self, s):
        # greedy
        if np.random.rand() > self._epsilon.value():
            # print("greedy action")
            s = Variable(torch.from_numpy(s).cuda())
            s = s.unsqueeze(0).float()
            self._action = self._Q(s).squeeze().cpu().data.numpy()
            return self._action.argmax()
        # explore
        else:
            target = np.random.randint(0, self._screen_size, size=2)
            return target[0] + target[1]

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
                obs = self.reset()[0]
                episode_length = 0
                # Get initial state
                state = obs.observation.feature_screen.player_relative #flatten dit
                print(self.env.observation_spec())
                print(self.env.observation_spec()[0].feature_screen[1])
                raise Exception("Implemented until this")
                state = torch.flatten(obs.observation.feature_screen)
                print(f'State shape: {state.shape}')
                state_mem = state
                state = torch.tensor(state, dtype=torch.float, device=device)
                total_episode_reward = 1

                raise Exception("Implemented until this")

                # A step in an episode
                while self.step < self.max_steps:
                    self.step += 1
                    episode_length += 1

                    # Choose action
                    prob_a = self.policy_network.pi(state)
                    action = torch.distributions.Categorical(prob_a).sample().item()

                    # Act
                    new_state, reward, terminal, _ = self.env.step(action)
                    new_state_mem = new_state
                    new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                    reward = -1 if terminal else reward

                    self.add_memory(state_mem, action, reward/10.0, new_state_mem, terminal, prob_a[action].item())

                    state = new_state
                    state_mem = new_state_mem
                    total_episode_reward += reward

                    if terminal:
                        reward_history.append(total_episode_reward)
                        avg_reward.append(sum(reward_history[-10:])/10.0)

                        self.finish_path(self.step)

                        print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                            'loss: %.4f, lr: %.4f' % (episode, step, episode_length, total_episode_reward, self.loss,
                                                        self.scheduler.get_last_lr()[0]))

                        self.env.reset()

                        break

                if episode % self.update_freq == 0:
                    for _ in range(self.k_epoch):
                        self.update_network()

                if episode % self.plot_every == 0:
                    plot_graph(reward_history, avg_reward)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(e)
        finally:
            self.env.close()
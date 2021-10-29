import numpy as np
from utils.DataManager import DataManager
from pysc2.lib import actions
from pysc2.lib import features

_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
FUNCTIONS = actions.FUNCTIONS
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

def _xy_locs(mask):
  """Mask should be a set of bools from comparison with a feature layer."""
  y, x = mask.nonzero()
  return list(zip(x, y))

class Agent:
    def __init__(self, env, max_episodes, store_obs):
        self.env = env
        self.max_episodes = max_episodes
        self.store_obs = store_obs
        self.map = "MoveToBeacon"
    
    def run_agent(self):
        print("Running Scripted agent on MoveToBeacon")
        if self.store_obs:
            print("Storing observations")
            self.data_manager = DataManager(observation_sub_dir=f'env_pysc2/observations/{self.map}')
            self.data_manager.create_observation_file()

        for eps in range(self.max_episodes):
            self.env.reset()
            select_friendly = self.select_friendly_action()
            obs = self.env.step([select_friendly])[0]

            state = obs.observation.feature_screen.player_relative
            if self.store_obs: self.data_manager.store_observation(state.flatten())

            while not obs.last():
                act = self.get_env_action(obs)
                obs = self.env.step([act])[0]
                new_state = obs.observation.feature_screen.player_relative
                if self.store_obs: self.data_manager.store_observation(new_state.flatten())

        self.env.close()

    def get_env_action(self, obs):
        player_relative = obs.observation.feature_screen.player_relative
        beacon = _xy_locs(player_relative == _PLAYER_NEUTRAL)
        beacon_center = np.mean(beacon, axis=0).round()
        return FUNCTIONS.Move_screen("now", beacon_center)
    
    def select_friendly_action(self):
        return FUNCTIONS.select_army("select")
    
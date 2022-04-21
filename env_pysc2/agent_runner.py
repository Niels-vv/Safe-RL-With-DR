#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run an agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from dqn.dqn import Agent as DQNAgent
from autoencoder.ae_agent import AEAgent
from principal_component_analysis.pca_agent import PCAAgent
from utils.DataManager import DataManager
from env_pysc2.models import policy_network, deep_mdp_encoder
from env_pysc2.models import ae_encoder, ae_decoder
from env_pysc2.ae import get_ae_config
from env_pysc2.pca import get_pca_config
from env_pysc2.env_wrapper import EnvWrapper
from env_pysc2.Hyperparameters import config

FLAGS = flags.FLAGS
flags.DEFINE_string("strategy", "dqn", "Which RL strategy to use.")
flags.DEFINE_string("variant", "base", "Whether to use VAE, PCA, or none.")
flags.DEFINE_bool("store_obs", False, "Whether to store observations.")
flags.DEFINE_bool("train", True, "Whether we are training or evaluating.")
flags.DEFINE_bool("load_policy", False, "Whether to load an existing policy network.")
flags.DEFINE_bool("train_ae_online", False, "Whether to use train ae online.") # TODO do this differently
flags.DEFINE_integer("max_episodes", 500, "Total episodes.")

flags.DEFINE_bool("render", True, "Whether to render with pygame.")
point_flag.DEFINE_point("feature_screen_size", "32",
                        "Resolution for screen feature layers.")
point_flag.DEFINE_point("feature_minimap_size", "64",
                        "Resolution for minimap feature layers.")
point_flag.DEFINE_point("rgb_screen_size", None,
                        "Resolution for rendered screen.")
point_flag.DEFINE_point("rgb_minimap_size", None,
                        "Resolution for rendered minimap.")
flags.DEFINE_enum("action_space", None, sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access
                  "Which action space to use. Needed if you take both feature "
                  "and rgb observations.")
flags.DEFINE_bool("use_feature_units", False,
                  "Whether to include feature units.")
flags.DEFINE_bool("use_raw_units", False,
                  "Whether to include raw units.")
flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")
flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")

flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
                    "Which agent to run, as a python path to an Agent class.")
flags.DEFINE_string("agent_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 1's race.")
flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
flags.DEFINE_string("agent2_name", None,
                    "Name of the agent in replays. Defaults to the class name.")
flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
                  "Agent 2's race.")
flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
                  "If agent2 is a built-in Bot, it's strength.")
flags.DEFINE_enum("bot_build", "random", sc2_env.BotBuild._member_names_,  # pylint: disable=protected-access
                  "Bot's build strategy.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

flags.DEFINE_string("map", "MoveToBeacon", "Name of a map to use.")
flags.DEFINE_bool("battle_net_map", False, "Use the battle.net map version.")
flags.mark_flag_as_required("map")

agent_class_name = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_thread(players, map_name, visualize):
  """Run one thread worth of the environment with agents."""
  global agent_class_name
  with sc2_env.SC2Env(
      map_name=map_name,
      battle_net_map=FLAGS.battle_net_map,
      players=players,
      agent_interface_format=sc2_env.parse_agent_interface_format(
          feature_screen=FLAGS.feature_screen_size,
          feature_minimap=FLAGS.feature_minimap_size,
          rgb_screen=FLAGS.rgb_screen_size,
          rgb_minimap=FLAGS.rgb_minimap_size,
          action_space=FLAGS.action_space,
          use_feature_units=FLAGS.use_feature_units,
          use_raw_units=FLAGS.use_raw_units),
      step_mul=FLAGS.step_mul,
      game_steps_per_episode=FLAGS.game_steps_per_episode,
      disable_fog=FLAGS.disable_fog,
      visualize=visualize) as env:
    env = available_actions_printer.AvailableActionsPrinter(env)
    run_agent(env)
    if FLAGS.save_replay:
      env.save_replay(agent_class_name)

# TODO remove PPO
def run_agent(env):
    global agent_class_name

    env = EnvWrapper(env, device)

    results_path = f'env_pysc2/results/dqn/{FLAGS.map}'                                     # Path for final results, relative to Package root
    intermediate_results_path = f'../drive/MyDrive/Thesis/Code/PySC2/{FLAGS.map}/Results'   # Path for intermediate results relative to package root. Using Google drive in case colab disconnects
    observations_path = f'../drive/MyDrive/Thesis/Code/PySC2/{FLAGS.map}/Observations'

    data_manager = DataManager(observation_sub_dir = observations_path, results_sub_dir=results_path, intermediate_results_sub_dir=intermediate_results_path)
    if FLAGS.store_obs:
        FLAGS.train = False
    if FLAGS.train:
        data_manager.create_results_files()
    encoder = None          # DeepMDP encoder
    conv_last = True        # Whether last layer in policy network is conv layer; needed for training the network

    if FLAGS.store_obs:
        # TODO scripted maken in env
        mlp = policy_network(dim_red=False)    # DDQN policy network. Not used in scripted agent
        agent = DQNAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp=False, train=False)
        total_obs = 240000
        obs = np.empty((total_obs, 32, 32),dtype=np.float32)
        agent.store_observations(total_obs = total_obs, observations = obs, skip_frame = 1)
    else:
        if FLAGS.variant.lower() in ["scripted"]:
            #TODO
            mlp = policy_network(dim_red=False)    # DDQN policy network. Not used in scripted agent
            agent_class_name = BeaconAgent.__name__
        if FLAGS.variant.lower() in ["base"]:
            deep_mdp = False
            mlp = policy_network(dim_red=False)    # DDQN policy network
            agent = DQNAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train)
        elif FLAGS.variant.lower() in ["pca"]:
            deep_mdp = False
            mlp = policy_network(dim_red=True)    # DDQN policy network
            pca_path = f'env_pysc2/results_pca/{FLAGS.map}' 
            pca_config = get_pca_config(pca_path)
            agent = PCAAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train, pca_config)
        elif FLAGS.variant.lower() in ["ae"]:
            deep_mdp = False
            mlp = policy_network(dim_red=True)    # DDQN policy network
            ae_path = f'env_pysc2/results_ae/{FLAGS.map}' 
            ae_config = get_ae_config(ae_path)
            agent = AEAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train, FLAGS.train_ae_online, ae_encoder, ae_decoder, ae_config)
        elif FLAGS.variant.lower() in ["deepmdp", "deep_mdp"]:
            #TODO Reduce_dim moet op false staan voor env_wrapper
            mlp = policy_network(dim_red=True)    # DDQN policy network
            deep_mdp = True
            agent = DQNAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train)
        else:
          raise NotImplementedError
    
    if not FLAGS.store_obs:
        if FLAGS.load_policy:
            load_policy(agent, results_path)
        agent.run_agent()

def load_policy(agent, results_path):
    checkpoint = DataManager.get_network(results_path, "policy_network.pt", device)
    agent.load_policy_checkpoint(checkpoint)
    if agent.train:
        agent.fill_buffer()

def main(unused_argv):
    """Run an agent."""
    if FLAGS.trace:
        stopwatch.sw.trace()
    elif FLAGS.profile:
        stopwatch.sw.enable()

    map_inst = maps.get(FLAGS.map)

    players = []
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent_race],
                                FLAGS.agent_name or agent_name))
    if map_inst.players >= 2:
        if FLAGS.agent2 == "Bot":
            players.append(sc2_env.Bot(sc2_env.Race[FLAGS.agent2_race],
                                    sc2_env.Difficulty[FLAGS.difficulty],
                                    sc2_env.BotBuild[FLAGS.bot_build]))
        else:
            agent_module, agent_name = FLAGS.agent2.rsplit(".", 1)
            players.append(sc2_env.Agent(sc2_env.Race[FLAGS.agent2_race],
                                        FLAGS.agent2_name or agent_name))

    run_thread(players, FLAGS.map, FLAGS.render)

    if FLAGS.profile:
        print(stopwatch.sw)


def entry_point():  # Needed so setup.py scripts work.
    app.run(main)


if __name__ == "__main__":
    app.run(main)

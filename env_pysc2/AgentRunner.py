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

import importlib
import threading

from absl import app
from absl import flags
from future.builtins import range  # pylint: disable=redefined-builtin

from pysc2 import maps
from pysc2.env import available_actions_printer
from pysc2.env import sc2_env
from pysc2.lib import point_flag
from pysc2.lib import stopwatch

from env_pysc2.ppo_variants.ppo_base import AgentLoop as PPOBaseAgent
from env_pysc2.dqn_variants.dqn_base import AgentLoop as DQNBaseAgent
from env_pysc2.vae_agent import get_agent as get_vae_agent
from env_pysc2.pca_agent import get_agent as get_pca_agent
from env_pysc2.scripted_beacon_agent import Agent as BeaconAgent

FLAGS = flags.FLAGS
flags.DEFINE_string("strategy", "dqn", "Which RL strategy to use.")
flags.DEFINE_string("variant", "base", "Whether to use VAE, PCA, or none.")
flags.DEFINE_bool("shield", False, "Whether to use shielding.")
flags.DEFINE_bool("store_obs", False, "Whether to store observations.")
flags.DEFINE_bool("train", True, "Whether we are training or evaluating.")
flags.DEFINE_bool("train_component", False, "Whether to train a sub component, like PCA or VAE, on stored observations.")
flags.DEFINE_bool("load_policy", False, "Whether to load an existing policy network.")
flags.DEFINE_integer("max_episodes", 500, "Total episodes.")
flags.DEFINE_integer("max_agent_steps", 1000, "Total agent steps.")

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
    agent = get_agent(env)
    agent.run_agent()
    if FLAGS.save_replay:
      env.save_replay(agent_class_name)

def get_agent(env):
  global agent_class_name

  if FLAGS.variant.lower() in ["scripted"]:
    FLAGS.save_replay = False
    return BeaconAgent(env, FLAGS.max_episodes, FLAGS.store_obs)
  elif FLAGS.variant.lower() in ["base"]:
    if FLAGS.strategy.lower() in ["dqn"]:
      agent_class_name = DQNBaseAgent.__name__
      return DQNBaseAgent(env, FLAGS.shield, FLAGS.max_agent_steps, FLAGS.max_episodes, FLAGS.train, FLAGS.map, FLAGS.load_policy)
    else:
      agent_class_name = PPOBaseAgent.__name__
      return PPOBaseAgent(env, FLAGS.shield, FLAGS.max_agent_steps, FLAGS.max_episodes, FLAGS.train, FLAGS.map, FLAGS.load_policy)
  elif FLAGS.variant.lower() in ["pca"]:
    agent_class_name, agent = get_pca_agent(FLAGS.strategy.lower(), env, FLAGS.shield, FLAGS.max_agent_steps, FLAGS.max_episodes, FLAGS.train, FLAGS.train_component, FLAGS.map, FLAGS.load_policy)
    return agent
  elif FLAGS.variant.lower() in ["vae"]:
    agent_class_name, agent = get_vae_agent(FLAGS.strategy.lower(), env, FLAGS.shield, FLAGS.max_agent_steps, FLAGS.max_episodes, FLAGS.train, FLAGS.train_component, FLAGS.map, FLAGS.load_policy)
    return agent
  else:
    raise NotImplementedError



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

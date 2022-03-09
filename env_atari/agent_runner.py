import torch
from absl import app, flags
from dqn.dqn import Agent as DQNAgent
from utils.DataManager import DataManager

FLAGS = flags.FLAGS
flags.DEFINE_string("strategy", "dqn", "Which RL strategy to use.")
flags.DEFINE_string("variant", "base", "Whether to use VAE, PCA, or none.")
flags.DEFINE_bool("train", True, "Whether we are training or evaluating.")
flags.DEFINE_bool("load_policy", False, "Whether to load an existing policy network.")
flags.DEFINE_bool("train_ae_online", False, "Whether to use train ae online.") # TODO do this differently
flags.DEFINE_integer("max_episodes", 500, "Total episodes.")
flags.DEFINE_integer("max_agent_steps", 1000, "Total agent steps.")


data_manager = DataManager(results_sub_dir=f'Atari/results/dqn/{game}')
mlp = None          # DQN Q* network
conv_last = False    # Whether last layer in mlp is conv layer
encoder = None      # DeepMDP encoder
deep_mdp = False
if deep_mdp:
  encoder = not None

def main(unused_argv):
    env = None
    agent = DQNAgent(env, FLAGS.max_steps, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deepmdp)

if __name__ == '__main__':
  app.run(main)
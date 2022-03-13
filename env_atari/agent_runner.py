import gym, torch
from absl import app, flags
from dqn.dqn import Agent as DQNAgent
from utils.DataManager import DataManager
from env_atari.models import policy_network, deep_mdp_encoder
from env_atari.env_wrapper import EnvWrapper
from env_atari.Hyperparameters import config

FLAGS = flags.FLAGS
flags.DEFINE_string("strategy", "dqn", "Which RL strategy to use.")
flags.DEFINE_string("variant", "base", "Whether to use baseline, AE, PCA, or DeepMDP.")
flags.DEFINE_bool("train", True, "Whether we are training or evaluating.")
flags.DEFINE_bool("load_policy", False, "Whether to load an existing policy network.")
flags.DEFINE_bool("train_ae_online", False, "Whether to use train ae online.") # TODO do this differently
flags.DEFINE_integer("max_episodes", 5000, "Total episodes.")
#flags.DEFINE_integer("max_agent_steps", 1000, "Total agent steps.")
flags.DEFINE_string("map", "ALE/Pong-v5", "OpenAI name of the game/env to use.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp = None                                    # Q* network
conv_last = False                             # Whether last layer in mlp is conv layer
encoder = None                                # DeepMDP encoder
deep_mdp = False
if deep_mdp:
    encoder = not None # TODO

def main(unused_argv):
    results_path = f'Atari/results/dqn/{FLAGS.map}'                                         # Path for final results, relative to Package root
    intermediate_results_path = f'../drive/MyDrive/Thesis/Code/Atari/{FLAGS.map}/Results'   # Path for intermediate results relative to package root. Using Google drive in case colab disconnects

    data_manager = DataManager(results_sub_dir=results_path, intermediate_results_sub_dir=intermediate_results_path)

    if FLAGS.train:
        data_manager.create_results_files()
    env = EnvWrapper(gym.make(FLAGS.map), device)
    input_shape = (4,84,84) # Shape of mlp input image: CxHxW
    mlp = policy_network(input_shape, env.env.action_space.n)
    encoder = deep_mdp_encoder if deep_mdp else None

    if FLAGS.variant.lower() in ["base"]:
        agent = DQNAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train)
    elif FLAGS.variant.lower() in ["pca"]:
        pass
    elif FLAGS.variant.lower() in ["ae"]:
        pass
    elif FLAGS.variant.lower() in ["deepmdp", "deep_mdp"]:
        pass

    agent.run_agent()

if __name__ == '__main__':
    app.run(main)
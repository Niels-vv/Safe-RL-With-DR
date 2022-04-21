import torch
import numpy as np
from absl import app, flags
from dqn.dqn import Agent as DQNAgent
from autoencoder.ae_agent import AEAgent
from principal_component_analysis.pca_agent import PCAAgent
from utils.DataManager import DataManager
from env_atari.models import policy_network, deep_mdp_encoder
from env_atari.models import ae_encoder, ae_decoder
from env_atari.ae import get_ae_config
from env_atari.pca import get_pca_config
from env_atari.env_wrapper import EnvWrapper
from env_atari.Hyperparameters import config
from env_atari.atari_wrappers import make_env

FLAGS = flags.FLAGS
flags.DEFINE_string("strategy", "dqn", "Which RL strategy to use.")
flags.DEFINE_string("variant", "base", "Whether to use baseline, AE, PCA, or DeepMDP.")
flags.DEFINE_bool("train", True, "Whether we are training or evaluating.")
flags.DEFINE_bool("load_policy", False, "Whether to load an existing policy network.")
flags.DEFINE_bool("train_ae_online", False, "Whether to use train ae online.") # TODO do this differently
flags.DEFINE_integer("max_episodes", 400, "Total episodes.")
#flags.DEFINE_integer("max_agent_steps", 1000, "Total agent steps.")
flags.DEFINE_string("map", "PongNoFrameskip-v4", "OpenAI name of the game/env to use.")
flags.DEFINE_bool("store_obs", False, "Whether to store observations.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp = None                                    # Q* network
conv_last = False                             # Whether last layer in mlp is conv layer
encoder = None                                # DeepMDP encoder
deep_mdp = False
if deep_mdp:
    encoder = not None # TODO

def main(unused_argv):
    results_path = f'env_atari/results/dqn/{FLAGS.map}'                                     # Path for final results, relative to Package root
    intermediate_results_path = f'../drive/MyDrive/Thesis/Code/Atari/{FLAGS.map}/Results'   # Path for intermediate results relative to package root. Using Google drive in case colab disconnects
    observations_path = f'../drive/MyDrive/Thesis/Code/Atari/{FLAGS.map}/Observations'

    data_manager = DataManager(observation_sub_dir = observations_path, results_sub_dir=results_path, intermediate_results_sub_dir=intermediate_results_path)
    if FLAGS.store_obs:
        FLAGS.train = False
    if FLAGS.train:
        data_manager.create_results_files()
    env = make_env(FLAGS.map)
    env.seed(0)
    env = EnvWrapper(env, device)
    input_shape = (4,84,84) # Shape of mlp input image: CxHxW
    mlp = policy_network(input_shape, env.env.action_space.n)
    encoder = None

    if FLAGS.store_obs:
        agent = DQNAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, train=False)
        load_policy(agent, results_path)
        agent.epsilon._value = 0.2 # 0.2 randomness when choosing action to prevent too similar episodes
        agent.epsilon.isTraining = True
        total_obs = 100000
        obs = np.empty((total_obs, input_shape[0], input_shape[1], input_shape[2]),dtype=np.float32)
        agent.store_observations(total_obs = total_obs, observations = obs, skip_frame = 50)
    else:
        if FLAGS.variant.lower() in ["base"]:
            agent = DQNAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train)
        elif FLAGS.variant.lower() in ["pca"]:
            pca_path = f'env_atari/results_pca/{FLAGS.map}' 
            pca_config = get_pca_config(pca_path)
            input_shape = (4,42,42) # Shape of mlp input image: CxHxW
            mlp = policy_network(input_shape, env.env.action_space.n)
            agent = PCAAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train, pca_config)
        elif FLAGS.variant.lower() in ["ae"]:
            ae_path = f'env_atari/results_ae/{FLAGS.map}' 
            ae_config = get_ae_config(ae_path)
            input_shape = (4,42,42) # Shape of mlp input image: CxHxW
            mlp = policy_network(input_shape, env.env.action_space.n)
            agent = AEAgent(env, config, device, FLAGS.max_episodes, data_manager, mlp, conv_last, encoder, deep_mdp, FLAGS.train, FLAGS.train_ae_online, ae_encoder, ae_decoder, ae_config)
        elif FLAGS.variant.lower() in ["deepmdp", "deep_mdp"]:
            pass
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

if __name__ == '__main__':
    app.run(main)
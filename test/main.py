import gym
from test.agent import Agent
from test.atari_wrappers import make_env
import time

# ---------------------------------Parameters----------------------------------

DQN_HYPERPARAMS = {
    'eps_start': 1,
    'eps_end': 0.02,
    'eps_decay': 10 ** 5,
    'buffer_size': 15000,
    'buffer_minimum': 10001,
    'learning_rate': 5e-5,
    'gamma': 0.99,
    'n_iter_update_nn': 1000,
    'multi_step': 2,
    'double_dqn': True,
    'dueling': False
}

ENV_NAME = "PongNoFrameskip-v4"
RECORD = False
MAX_GAMES = 300
DEVICE = 'cuda'
BATCH_SIZE = 32

# ------------------------Create enviroment and agent--------------------------
env = make_env("PongNoFrameskip-v4")  # gym.make("PongNoFrameskip-v4")
# For recording few seelcted episodes. 'force' means overwriting earlier recordings
obs = env.reset()
# Create agent that will learn
agent = Agent(env, hyperparameters=DQN_HYPERPARAMS, device=DEVICE, writer=None, max_games=MAX_GAMES, tg_bot=None)
# --------------------------------Learning-------------------------------------
num_games = 0
while num_games < MAX_GAMES:
    # Select one action with e-greedy policy and observe s,a,s',r and done
    action = agent.select_eps_greedy_action(obs)
    # Take that action and observe s, a, s', r and done
    new_obs, reward, done, _ = env.step(action)
    # Add s, a, s', r to buffer B
    agent.add_to_buffer(obs, action, new_obs, reward, done)
    # Sample a mini-batch from buffer B if B is large enough. If not skip until it is.
    # Use that mini-batch to improve NN value function approximation
    agent.sample_and_improve(BATCH_SIZE)

    obs = new_obs
    if done:
        num_games = num_games + 1
        agent.print_info()
        agent.reset_parameters()
        obs = env.reset()

gym.wrappers.Monitor.close(env)
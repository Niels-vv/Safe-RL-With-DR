import torch
import matplotlib.pyplot as plt
import pandas as pd
from ppo.PPO import Agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AgentLoop(Agent):
    def __init__(self, env, max_steps, max_episodes, test=False):
        observations_space = env.observations_spec()
        action_space = env.action_spec()
        super(AgentLoop, self).__init__(env, observations_space, action_space, max_steps, max_episodes, test)

    def run_loop(self):
        episode = 0
        step = 0
        reward_history = []
        avg_reward = []
        solved = False

        # A new episode
        while not solved:
            start_step = step
            episode += 1
            episode_length = 0

            # Get initial state
            state, reward, action, terminal = self.new_random_game()
            state_mem = state
            state = torch.tensor(state, dtype=torch.float, device=device)
            total_episode_reward = 1

            # A step in an episode
            while not solved:
                step += 1
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
                    episode_length = step - start_step
                    reward_history.append(total_episode_reward)
                    avg_reward.append(sum(reward_history[-10:])/10.0)

                    self.finish_path(episode_length)

                    if len(reward_history) > 100 and sum(reward_history[-100:-1]) / 100 >= 195:
                        solved = True

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

        self.env.close()

def plot_graph(reward_history, avg_reward):
    df = pd.DataFrame({'x': range(len(reward_history)), 'Reward': reward_history, 'Average': avg_reward})
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')

    plt.plot(df['x'], df['Reward'], marker='', color=palette(1), linewidth=0.8, alpha=0.9, label='Reward')
    # plt.plot(df['x'], df['Average'], marker='', color='tomato', linewidth=1, alpha=0.9, label='Average')

    # plt.legend(loc='upper left')
    plt.title("CartPole", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)

    plt.savefig('score.png')
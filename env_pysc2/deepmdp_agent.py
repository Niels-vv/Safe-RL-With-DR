from env_pysc2.dqn_variants.dqn_base import AgentLoop as DQNAgent

class DQNAgentLoop(DQNAgent):
    def __init__(self, env, shield, max_steps, max_episodes, train, map_name, load_policy):
        latent_space = 16*16 # TODO Magic number
        print(f'DeepMDP latent space: {latent_space}')
        super(DQNAgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, map_name, load_policy, latent_space)
        self.deepmdp = True
        self.setup_deepmdp()

    def run_agent(self):
        super(DQNAgentLoop, self).run_agent()
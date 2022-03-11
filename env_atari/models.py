import torch
import torch.nn as nn

seed = 0
torch.manual_seed(seed)

c_hid = 32
act_fn = nn.GELU

# Q network for agent
policy_network = nn.Sequential(
                    nn.Conv2d(1, c_hid, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(c_hid, 1, kernel_size=3, stride=1, padding=1)
                )

# Encoder for mlp when using Deep_MDP
deep_mdp_encoder = nn.Sequential(
                    nn.Conv2d(1, c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                    act_fn(),
                    nn.Conv2d(c_hid, 1, kernel_size=3, padding=1, stride=1),
                    act_fn()
                )
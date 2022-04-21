import torch
import torch.nn as nn

seed = 0
torch.manual_seed(seed)

# Q network for agent.
def policy_network(dim_red = False):
    c_hid = 32
    if dim_red: # Using dimensionality reduction, so input is 16 x 16 instead of 32 x 32; upscale input tot 32 x 32
        mlp =  nn.Sequential(
                nn.ConvTranspose2d(1, c_hid, kernel_size=3, stride=2, padding=1, output_padding = 1), # 16 x 16 => 32 x 32
                nn.ReLU(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(c_hid, 1, kernel_size=3, stride=1, padding=1)
            )
    else:
        mlp =  nn.Sequential(
                nn.ConvTranspose2d(1, c_hid, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(c_hid, 1, kernel_size=3, stride=1, padding=1)
            )
    return mlp

ae_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2),   # 32x32 => 16x16
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
        )

ae_decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, kernel_size=3, output_padding=1, padding=1, stride=2),    # 16x16 => 32x32
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1)
        )

# Encoder for mlp when using Deep_MDP
deep_mdp_encoder = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                    nn.GELU(),
                    nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
                    nn.GELU()
                )

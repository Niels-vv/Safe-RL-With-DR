import torch
import torch.nn as nn

seed = 0
torch.manual_seed(seed)

c_hid_first = 32
c_hid_second = 64
act_fn = nn.GELU

# Q network for agent.
def policy_network(input_shape, linear_output):
    input_channels = input_shape[0]
    structure = [(c_hid_first, 8, 4, 0), (c_hid_second, 4, 2, 0), (c_hid_second, 3, 1, 0)] # structure of each conv layer: (out_channels, kernel_size, stride, padding)
    linear_input = get_conv_output_shape_flattened(input_shape, structure)
    return nn.Sequential(
                nn.Conv2d(input_channels, c_hid_first, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(c_hid_first, c_hid_second, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(c_hid_second, c_hid_second, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(linear_input, 512),
                nn.ReLU(),
                nn.Linear(512, linear_output)
            )

# Encoder for mlp when using Deep_MDP
deep_mdp_encoder = nn.Sequential(
                    nn.Conv2d(1, c_hid_first, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
                    act_fn(),
                    nn.Conv2d(c_hid_first, 1, kernel_size=3, padding=1, stride=1),
                    act_fn()
                )

def get_conv_output_shape_flattened(input_shape, structure):
    """
        Input_shape: (Height, Width) of input image
        structure: List containing tuple (out_channels, kernel_size, stride, padding) per conv layer
    """
    def get_layer_output(input, layer_structure):
        return int((input + 2*layer_structure[3] - layer_structure[1] - 2) / layer_structure[2]) + 1
        
    w = input_shape[0]
    h = input_shape[1]
    c = structure[-1][0]
    for layer in structure:
        w = get_layer_output(w, layer)
        h = get_layer_output(h, layer)
    return w * h * c

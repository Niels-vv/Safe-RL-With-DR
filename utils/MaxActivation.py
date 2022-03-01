import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile
from scipy.ndimage import gaussian_filter

# https://github.com/Nguyen-Hoa/Activation-Maximization


"""
Create a hook into target layer
    Example to hook into classifier 6 of Alexnet:
        alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
"""
def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook

"""
Reguarlizer, crop by absolute value of pixel contribution
"""
def abs_contrib_crop(img, threshold=0):

    abs_img = torch.abs(img)
    smalls = abs_img < percentile(abs_img, threshold)
    
    return img - img*smalls

"""
Regularizer, crop if norm of pixel values below threshold
"""
def norm_crop(img, threshold=0):

    norm = torch.norm(img, dim=0)
    norm = norm.numpy()

    # Create a binary matrix, with 1's wherever the pixel falls below threshold
    smalls = norm < percentile(norm, threshold)
    smalls = tile(smalls, (3,1,1))

    # Crop pixels from image
    crop = img - img*smalls
    return crop

"""
Optimizing Loop
    Dev: maximize layer vs neuron
"""
def act_max(network, 
    input, 
    layer_activation, 
    layer_name, 
    unit, 
    steps=50, 
    alpha=torch.tensor(100),
    L2_Decay=True, 
    theta_decay=0.1,
    Gaussian_Blur=True,
    theta_every=4,
    theta_width=1,
    Norm_Crop=True,
    theta_n_crop=30,
    Contrib_Crop=True,
    theta_c_crop=30,
    ):

    best_activation = -float('inf')
    best_img = input

    for k in range(steps):

        input.retain_grad() # non-leaf tensor
        # network.zero_grad()
        
        # Propogate image through network,
        # then access activation of target layer
        network(input)
        layer_out = layer_activation[layer_name]

        # compute gradients w.r.t. target unit,
        # then access the gradient of input (image) w.r.t. target unit (neuron) 
        layer_out[0][unit].backward(retain_graph=True)
        img_grad = input.grad

        # Gradient Step
        # input = input + alpha * dimage_dneuron
        input = torch.add(input, torch.mul(img_grad, alpha))

        # regularization does not contribute towards gradient
        """
        DEV:
            Detach input here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                input = torch.mul(input, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and k % theta_every is 0:
                temp = input.squeeze(0)
                temp = temp.detach().numpy()
                for channel in range(3):
                    cimg = gaussian_filter(temp[channel], theta_width)
                    temp[channel] = cimg
                temp = torch.from_numpy(temp)
                input = temp.unsqueeze(0)

            # Regularization: Clip Norm
            if Norm_Crop:
                input = norm_crop(input.detach().squeeze(0), threshold=theta_n_crop)
                input = input.unsqueeze(0)

            # Regularization: Clip Contribution
            if Contrib_Crop:
                input = abs_contrib_crop(input.detach().squeeze(0), threshold=theta_c_crop)
                input = input.unsqueeze(0)

        input.requires_grad_(True)

        # Keep highest activation
        if best_activation < layer_out[0][unit]:
            best_activation = layer_out[0][unit]
            best_img = input

    return best_img


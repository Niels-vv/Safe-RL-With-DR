import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile
from scipy.ndimage import gaussian_filter

# https://github.com/Nguyen-Hoa/Activation-Maximization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    img, 
    layer_activation, 
    gradients,
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
    gradients = None
    def get_gradient(grad):
        gradients = grad

    best_activation = -float('inf')
    best_img = img

    for k in range(steps):

        img.retain_grad() # non-leaf tensor
        # network.zero_grad()
        
        # Propogate image through network,
        # then access activation of target layer
        network(img)
        layer_out = layer_activation[layer_name]

        # compute gradients w.r.t. target unit,
        # then access the gradient of img (image) w.r.t. target unit (neuron) 
        #layer_out[0][unit].backward(retain_graph=True)
        loss = -layer_out[unit].mean()
        loss.register_hook(get_gradient) 
        loss.backward(retain_graph=True)
        #img_grad = img.grad
        gradients /= (torch.sqrt(torch.mean(
                torch.mul(gradients, gradients))) + 1e-5)
        # Gradient Step
        # img = img + alpha * dimage_dneuron
        print(gradients)
        img = torch.add(img, torch.mul(gradients, alpha))

        # regularization does not contribute towards gradient
        """
        DEV:
            Detach img here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                img = torch.mul(img, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and k % theta_every is 0:
                temp = img.squeeze(0)
                temp = temp.detach().numpy()
                for channel in range(3):
                    cimg = gaussian_filter(temp[channel], theta_width)
                    temp[channel] = cimg
                temp = torch.from_numpy(temp)
                img = temp.unsqueeze(0)

            # Regularization: Clip Norm
            if Norm_Crop:
                img = norm_crop(img.detach().squeeze(0), threshold=theta_n_crop)
                img = img.unsqueeze(0)

            # Regularization: Clip Contribution
            if Contrib_Crop:
                img = abs_contrib_crop(img.detach().squeeze(0), threshold=theta_c_crop)
                img = img.unsqueeze(0)

        img.requires_grad_(True)

        # Keep highest activation
        if best_activation < layer_out[0][unit]:
            best_activation = layer_out[0][unit]
            best_img = img

    return best_img


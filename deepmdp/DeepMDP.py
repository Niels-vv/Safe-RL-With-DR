import torch
from torch import nn
import torch.nn.functional as F

# Class to compute transition cost for DeepMDP
class TransitionAux(nn.Module):
    def __init__(self, device):
        super(TransitionAux, self).__init__()
        self.c_hid = 1
        self.action_dim = 32 * 32 # TODO pysc2 specific

        zero_pad = torch.nn.ZeroPad2d((0, 1, 0, 1)) # maintain input dimensionality.
        conv = torch.nn.Conv2d(in_channels = self.c_hid, out_channels = self.c_hid * self.action_dim,kernel_size = 2, stride = 1).to(device)
        self.network = torch.nn.Sequential(zero_pad, conv, F.relu())

    def compute_loss(self, embedding, embedding_next_observation, actions):
        """
        Compute loss between embedding of next observation and the predicted embedding of the next observation.
        :param embedding: The embedded observation used as an input for the latent transition network
        :param embedding_next_observation: The ground truth embedded next obervation
        :param actions: The actions that caused the embedded next_observations.
        :return: The mean squared error between the predicted and the ground truth embedding of the next observation.
        """
        preds = self.network(embedding)

        print(f'Auxiliary preds shape: {preds.shape}')

        batch_size = actions.size(0)
        # Reshape tensor: B x act * channels ... --> B x channels x ... x act
        preds = preds.unsqueeze(len(preds.size())).reshape(batch_size, self.c_hid, *preds.size()[2:4], self.action_dim)
        print(f'Auxiliary preds shape: {preds.shape}')

        loss_func = torch.nn.SmoothL1Loss()
        loss = 0
        for i, act in enumerate(actions):
            predicted__next_observation_embedding = preds[i, ..., int(act.item())].squeeze()
            ground_truth_embedding = embedding_next_observation[i, ...]
            assert(ground_truth_embedding.size() == predicted__next_observation_embedding.size())
            loss += loss_func(predicted__next_observation_embedding, ground_truth_embedding)
        return loss


def compute_deepmdp_loss(policy_network, auxiliary_objective, s, s_1, actions, state_embeds, new_states, penalty, device):
    loss = 0
    
    with torch.no_grad():
        next_state_embeds, _ = policy_network(s_1, return_deepmdp = True)
    loss += auxiliary_objective.compute_loss(state_embeds, next_state_embeds, actions)
    
    print(f'Loss after auxiliary: {loss}')

    
    with torch.no_grad():
        new_state_embeds, _ = policy_network(new_states, return_deepmdp = True)
        new_state_embeds = torch.from_numpy(new_state_embeds).to(device).float()

    gradient_penalty = 0
    gradient_penalty += compute_gradient_penalty(policy_network.encoder, s, new_states, device)
    print(f'Gradient penalty after encoder: {gradient_penalty}')
    gradient_penalty += compute_gradient_penalty(policy_network.mlp, state_embeds, new_state_embeds, device)
    print(f'Gradient penalty after dqn mlp: {gradient_penalty}')
    gradient_penalty += compute_gradient_penalty(auxiliary_objective.network, state_embeds, new_state_embeds, device)
    print(f'Gradient penalty after auxiliary: {gradient_penalty}')
    loss += penalty * gradient_penalty

# Helper function computing Wasserstein Generative Adversarial Network penalty
def compute_gradient_penalty(network, samples_a, samples_b, device):
        # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py

        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        batch_size = samples_a.size(0)
        alpha = torch.rand_like(samples_a)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * samples_a + ((1 - alpha) * samples_b))
        interpolated_obs = torch.autograd.Variable(interpolates, requires_grad=True)

        d_interpolates = network(interpolated_obs)
        grad = torch.ones(d_interpolates.size(), requires_grad=False).to(device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolated_obs,
            grad_outputs=grad,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(int(batch_size), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
import torch
from torch import nn

class TransitionAux(nn.Module):
    def __init__(self):
        super(TransitionAux, self).__init__()
        self.network = None

    def compute_loss(self, embedding, embedding_next_observation, actions):
        """
        Compute loss between embedding of next observation and the predicted embedding of the next observation.
        :param embedding: The embedded observation used as an input for the latent transition network
        :param embedding_next_observation: The ground truth embedded next obervation
        :param actions: The actions that caused the embedded next_observations.
        :return: The mean squared error between the predicted and the ground truth embedding of the next observation.
        """
        preds = self.network(embedding)
        batch_size = actions.size(0)
        # Reshape tensor: B x act * channels ... --> B x channels x ... x act
        preds = preds.unsqueeze(len(preds.size())).reshape(batch_size, self.input_dim, *preds.size()[2:4], self.action_dim)
        loss_func = torch.nn.SmoothL1Loss()
        loss = 0
        for i, act in enumerate(actions):
            predicted__next_observation_embedding = preds[i, ..., int(act.item())].squeeze()
            ground_truth_embedding = embedding_next_observation[i, ...]
            assert(ground_truth_embedding.size() == predicted__next_observation_embedding.size())
            loss += loss_func(predicted__next_observation_embedding, ground_truth_embedding)
        return loss
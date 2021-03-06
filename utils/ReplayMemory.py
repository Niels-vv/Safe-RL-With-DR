# adapted from https://github.com/whathelll/DeepRLBootCampLabs/tree/master/pytorch/utils
from collections import namedtuple
import random
import numpy as np


Transition = namedtuple("Transition", ["s", "a", "s_1", "r", "done"])


class ReplayMemory(object):
    def __init__(self, capacity, min_train_buffer):
        self.capacity = capacity
        self.memory = []
        self.min_train_buffer = min(min_train_buffer, capacity)
        self.position = 0

    def push(self, item):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        out = random.sample(self.memory, batch_size)
        batched = Transition(*zip(*out))
        s = np.array(list(batched.s))
        a = np.expand_dims(np.array(list(batched.a)), axis=1)
        s_1 = np.array(list(batched.s_1))
        r = np.expand_dims(np.array(list(batched.r)), axis=1)
        done = np.expand_dims(np.array(list(batched.done)), axis=1)
        return [s, a, s_1, r, done]

    def is_filled(self):
        return len(self.memory) >= self.capacity

    def ready_for_training(self):
        return len(self.memory) >= self.min_train_buffer

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.memory[i].__str__() + " \n")
        return "".join(result)

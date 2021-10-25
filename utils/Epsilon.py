# adapted from https://github.com/whathelll/DeepRLBootCampLabs/tree/master/pytorch/utils
import numpy as np
class Epsilon(object):
    def __init__(self, start=1.0, end=0.1, decay_steps=80000):
        self.start = start
        self.end = end
        #self._update_increment = update_increment
        self.decay_steps = decay_steps
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (start - end) + end
        self._value = self.start
        self.t = 0
        self.isTraining = True
    
    def increment(self):
        #self.value = max(self._end, self._value - self._update_increment*count)
        self._value = self.end if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        
    def value(self):
        if not self.isTraining:
            return 0.0
        else:
            return self._value

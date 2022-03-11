from abc import ABC, abstractmethod
 
class EnvWrapperAbstract(ABC):
    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def get_state(self, obs, reduce_dim, reduction_component, pca, ae):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_last_obs(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reduce_dim(self, reduction_component, pca, ae):
        pass

    @abstractmethod
    def add_obs_to_ae_batch(self):
        pass
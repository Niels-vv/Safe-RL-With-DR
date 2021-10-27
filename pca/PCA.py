import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCACompression:
    def __init__(self):
        self.fileNames = []
        self.pca_main = None
        self.pcaStatistic = PCA()
        self.scaler = StandardScaler()
        self.latent_space = None

    def create_pca(self, observations):
        self.scaler.fit(observations)
        df = self.scaler.transform(observations)
        self.pcaStatistic.fit(df)
        
    def update_pca(self, observations, dimensions):
        self.latent_space = dimensions
        df = self.scaler.transform(observations)
        self.pca_main = PCA(n_components=dimensions)
        self.pca_main.fit(df)

    def state_dim_reduction(self, observation):
        obs = self.scaler.transform([observation.cpu().numpy()])
        return torch.tensor(self.pca_main.transform(obs)[0], dtype=torch.float, device=device)

    def get_pca_dimension_info(self):
        return np.cumsum(self.pcaStatistic.explained_variance_ratio_) 
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
        self.df = None

    def create_pca(self, observations):
        print("Fitting the scalar...")
        self.scaler.fit(observations)
        print("Transforming the scalar...")
        self.df = self.scaler.transform(observations)
        print("Fitting statistics PCA...")
        self.pcaStatistic.fit(self.df)
        
    def update_pca(self, dimensions):
        self.latent_space = dimensions
        self.pca_main = PCA(n_components=dimensions)
        print(f'Fitting final PCA on latent space {dimensions}')
        self.pca_main.fit(self.df)

    def state_dim_reduction(self, observation):
        obs = self.scaler.transform([observation.cpu().numpy()])
        return torch.tensor(self.pca_main.transform(obs)[0], dtype=torch.float, device=device)

    def get_pca_dimension_info(self):
        return np.cumsum(self.pcaStatistic.explained_variance_ratio_) 
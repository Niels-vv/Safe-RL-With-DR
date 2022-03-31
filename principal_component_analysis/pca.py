import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA as PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCACompression:
    def __init__(self, scalar, latent_space):
        self.fileNames = []
        self.pca_main = None
        self.batch_size = 10000
        self.pcaStatistic = PCA(batch_size = self.batch_size)
        self.scaler = StandardScaler()
        self.use_scalar = scalar
        self.latent_space = latent_space
        self.df = None

    def create_pca(self, observations, get_statistics):
        if self.use_scalar:
            print("Fitting the scalar...")
            self.scaler.fit(observations)
            print("Transforming the scalar...")
            self.df = self.scaler.transform(observations)
        else:
            self.df = observations
        if get_statistics:
            print("Fitting statistics PCA...")
            self.pcaStatistic.fit(self.df)
        
    def update_pca(self):
        self.pca_main = PCA(n_components=self.latent_space, batch_size = self.batch_size)
        print(f'Fitting final PCA on latent space {self.latent_space}')
        self.pca_main.fit(self.df)

    def state_dim_reduction(self, observation):
        state = []
        for obs in observation:
            obs = obs.flatten().cpu().numpy()
            if self.use_scalar:
                obs = self.scaler.transform([obs])
            else:
                obs = [obs]
            state.append(self.pca_main.transform(obs)[0])
        return torch.tensor(state, dtype=torch.float, device=device)

    def get_pca_dimension_info(self):
        return np.cumsum(self.pcaStatistic.explained_variance_ratio_) 
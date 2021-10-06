import sys, csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from env_pysc2.ppo_variants.ppo_base import AgentLoop as Agent

class AgentLoop(Agent):
    def __init__(self, env, shield, max_steps, max_episodes, train, train_component, map_name):
        super(AgentLoop, self).__init__(env, shield, max_steps, max_episodes, train, False, map_name)

        self.train_component = train_component
        self.reduce_dim = not train_component
        self.pca = True

    def run_agent(self):
        self.store_obs = False
        if self.train_component:
            pass
        else:
            raise NotImplementedError
            self.data_manager = DataManager(results_sub_dir=f'env_pysc2/results/{self.map}')
            self.dim_reduction_component  = self.data_manager.get_pca()
            super(AgentLoop, self).run_agent()

class PCACompression:
    def __init__(self, pcaComponents, existingFile = None, fileNameBase="drive/MyDrive/Thesis/Code/RL_PCA/feature_data",  RGB = False):
        self.fileNames = []
        self.existingFile = existingFile
        self.fileNameBase = fileNameBase
        self.pca_main = PCA(n_components=pcaComponents)
        self.pcaStatistic = PCA()
        self.scaler = StandardScaler()
        self.RGB = RGB
        if ((existingFile is not None) and (existingFile is not "")):
            self.fileNames.append(existingFile)
        elif RGB:
            self.create_files(3)
        else:
            self.create_files()

    def create_pca(self):
        if self.RGB:
            #TODO
            dfR = pd.read_csv(self.fileNames[0])
            dfG = pd.read_csv(self.fileNames[1])
            dfB = pd.read_csv(self.fileNames[2])
        else:
            df = pd.read_csv(self.fileNames[0])
            self.scaler.fit(df)
            df = self.scaler.transform(df)
            self.pca_main.fit(df)
            self.pcaStatistic.fit(df)
            
    def update_pca(self, dimensions):
        df = pd.read_csv(self.fileNames[0])
        self.scaler.fit(df)
        df = self.scaler.transform(df)
        self.pca_main = PCA(n_components=dimensions)
        self.pca_main.fit(df)
        self.pcaStatistic.fit(df)

    def state_dim_reduction(self, observation):
        obs = self.scaler.transform([observation.cpu().numpy()])
        return self.pca_main.transform(obs)[0]

    def get_pca_dimension_info(self):
        return np.cumsum(self.pcaStatistic.explained_variance_ratio_) 
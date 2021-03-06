# (static) Methods for:
# Storing observations (in drive/MyDrive/Thesis/Code/PySC2/Observations/Observations.csv)
# Reading observations from file
# Writing results (rewards, steps and physcial time per episode and model setup) (in drive/MyDrive/Thesis/Code/PySC2/Results/fileName.csv and /fileName.txt)
# Creating plots from results file

import os.path, sys, csv, json, torch, pickle
import pandas as pd
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__)) # Path to DataManager file

'''Data manager class to handle files for storing results, state observations, pca and vae'''
class DataManager:
    def __init__(self, observation_sub_dir = None , results_sub_dir = None, intermediate_results_sub_dir = None):
        self.observations_path = f'{PATH}/../{observation_sub_dir}'
        self.observations_sub_dir = observation_sub_dir
        self.results_path = f'{PATH}/../{results_sub_dir}'
        self.intermediate_results_path = f'{PATH}/../{intermediate_results_sub_dir}'

        self.obs_file = None
        self.results_file = None
        self.setup_file = None
        self.variant_file = None
        self.policy_network_file = None
        self.ae_file = None
        self.intermediate_policy_network_file = None
        self.intermediate_results_file = None
        self.intermediate_ae_file = None


    '''Setup directories and files for storing state observations'''
    def create_observation_file(self):
        if not os.path.isdir(f'{self.observations_path}'): os.makedirs(self.observations_path)
        i = 1
        self.obs_file = f'{self.observations_path}/Observations_{i}.npy'
        while(os.path.isfile(self.obs_file)):
            i += 1
            self.obs_file = f'{self.observations_path}/Observations_{i}.npy'
        with open(self.obs_file, mode='w') as fp:
            pass

    '''Setup directories for storing AE results'''
    def create_dim_red_results_dirs(self, i=1):
        if not os.path.isdir(f'{self.results_path}'): os.makedirs(self.results_path)
        self.obs_file = f'{self.observations_path}/Observations_{i}.npy'
        return os.path.isfile(self.obs_file)

    '''Setup directories and files for storing results'''
    def create_results_files(self):
        if not os.path.isdir(self.results_path): os.makedirs(self.results_path)
        i = 1
        self.results_file = f'{self.results_path}/results_{i}.csv'
        while(os.path.isfile(self.results_file)):
            i += 1
            self.results_file = f'{self.results_path}/results_{i}.csv'
        with open(self.results_file, mode = 'w') as fp:
            pass
        self.setup_file = f'{self.results_path}/setup_{i}.json'
        self.variant_file = f'{self.results_path}/variant_{i}.json'
        self.policy_network_file = f'{self.results_path}/policy_network_{i}.pt'
        self.ae_file = f'{self.results_path}/ae_online_{i}.pt'

        if not os.path.isdir(self.intermediate_results_path): os.makedirs(self.intermediate_results_path)
        with open(f'{self.intermediate_results_path}/Results.csv', "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Epsilon", "Duration"])
        self.intermediate_policy_network_file = f'{self.intermediate_results_path}/policy_network.pt'
        self.intermediate_results_file = f'{self.intermediate_results_path}/Results.csv'
        self.intermediate_ae_file = f'{self.intermediate_results_path}/ae_online.pt'

    '''Write state observation to file, creating demo trajectories to use for training PCA and AE'''
    def store_observation(self, data):
        with open(self.obs_file, 'wb') as f:
            np.save(f,data)

    def get_observations(self):
        data = None
        with open(self.obs_file, 'rb') as f:
            data = np.load(self.obs_file)
        return data

    def store_dim_reduction_component(self, component, name):
        if not os.path.isdir(f'{self.results_path}'): os.makedirs(self.results_path)
        with open(f'{self.results_path}/{name}', 'wb') as fp:
          pickle.dump(component, fp)

    def store_network(self, checkpoint, name):
        if not os.path.isdir(f'{self.results_path}'): os.makedirs(self.results_path)
        torch.save(checkpoint, f'{self.results_path}/{name}')

    @staticmethod
    def get_network(rel_path, network_name, device):
        if device.type == "cpu":
            return torch.load(f'{PATH}/../{rel_path}/{network_name}', map_location="cpu")
        else:
            return torch.load(f'{PATH}/../{rel_path}/{network_name}')

    @staticmethod
    def get_component(rel_path, component_name):
        with open(f'{PATH}/../{rel_path}/{component_name}', 'rb') as fp:
            return pickle.load(fp)


    '''Store results of DQN after training; storing setup info, training results and policy network'''
    def write_results(self, eps, rewards, epsilons, durations, setup, variant, network_checkpoint, ae_checkpoint=None):
        rows = zip(eps, rewards, epsilons, durations)
        try:
            with open(self.results_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Reward", "Epsilon", "Duration"])
                for row in rows:
                    writer.writerow(row)
            
            with open(self.setup_file, 'w') as f:
                json.dump(setup, f)

            with open(self.variant_file, 'w') as f:
                json.dump(variant, f)
            
            torch.save(network_checkpoint, self.policy_network_file)
            if ae_checkpoint is not None: torch.save(ae_checkpoint, self.ae_file)

        except Exception as e:
            print("writing results failed")
            print(e)

    def write_intermediate_results(self, eps, rewards, durations, epsilons, network, ae_checkpoint):
        rows = zip(eps, rewards, epsilons, durations)
        try:
            with open(self.intermediate_results_file, "a") as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)

            torch.save(network, self.intermediate_policy_network_file)
            if ae_checkpoint is not None: torch.save(ae_checkpoint, self.intermediate_ae_file)
        except Exception as e:
            print("writing results failed")
            print(e)

    @staticmethod
    def load_setup_dict(filename):
        with open(filename) as f:
            return json.load(f)

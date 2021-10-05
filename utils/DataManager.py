# (static) Methods for:
# Storing observations (in drive/MyDrive/Thesis/Code/PySC2/Observations/Observations.csv)
# Reading observations from file
# Writing results (rewards, steps and physcial time per episode and model setup) (in drive/MyDrive/Thesis/Code/PySC2/Results/fileName.csv and /fileName.txt)
# Creating plots from results file

import os.path, sys, csv, json
#import pandas as pd
import numpy as np

PATH = os.path.dirname(os.path.realpath(__file__))

class DataManager:
    def __init__(self, observation_sub_dir = None , results_sub_dir = None):
        self.obs_sub_dir = observation_sub_dir
        self.results_sub_dir = results_sub_dir
        self.obs_file = None
        self.results_file = None
        self.setup_file = None

    def create_observation_file(self):
        observations_path = f'{PATH}/{self.obs_sub_dir}'
        if not os.path.isdir(f'{observations_path}'): os.mkdir(observations_path)
        observations_file = f'{observations_path}/Observations.csv'
        with open(observations_file, mode='w') as fp:
            pass
        self.obs_file = observations_file

    def create_results_files(self):
        results_path = f'{PATH}/{self.results_sub_dir}'
        if not os.path.isdir(f'{results_path}'): os.mkdir(results_path)
        i = 1
        results_file = f'{results_path}/results_{i}.csv'
        while(os.path.isfile(results_file)):
            i += 1
            results_file = f'{results_path}/results_{i}.csv'
        with open(self.results_file, mode = 'w') as fp:
            pass
        self.setup_file = f'{results_path}/setup_{i}.json'
        # TODO network weights path?


    def store_observation(self, data):
        if not (isinstance(data, list) or isinstance(data, np.ndarray)):
            raise TypeError
        try:
            with open(self.obs_file, mode='a') as fp:
                data_writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(data)
        except Exception as e:
            print("Writing data to csv failed.")
            print(e)

    @staticmethod
    def get_observations():
        pass

    def write_results(self, rewards, steps, duration, setup, network_weights):
        rows = zip(rewards, steps, duration)
        try:
            with open(self.results_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(["Rewards", "steps", "Duration"])
                for row in rows:
                    writer.writerow(row)
            
            with open(self.setup_file, 'w') as f:
                json.dump(setup, f)
            # TODO network weights opslaan
        except Exception as e:
            print("writing results failed")
            print(e)

    @staticmethod
    def show_plot():
        pass

    @staticmethod
    def load_setup_dict(filename):
        filename = results_path + '/' + filename
        with open(filename) as f:
            return json.load(f)

DataManager.store_observation([0,1,2,3])
DataManager.write_results("test",[0,1,2], [1,2,3],[3,4,5],{'r':2,'s':3})
print(DataManager.load_setup_dict(f'test_setup.json'))
#TODO: get observations. Test on colab/drive

# (static) Methods for:
# Storing observations (in drive/MyDrive/Thesis/Code/PySC2/Observations/Observations.csv)
# Reading observations from file
# Writing results (rewards, steps and physcial time per episode and model setup) (in drive/MyDrive/Thesis/Code/PySC2/Results/fileName.csv and /fileName.txt)
# Creating plots from results file

import os.path, sys, csv, json
#import pandas as pd
import numpy as np

#create directories and files for observations and results
PATH = os.path.dirname(os.path.realpath(__file__))
observations_path = f'{PATH}/../../Observations'
results_path = f'{PATH}/../../Results'
if not os.path.isdir(f'{observations_path}'): os.mkdir(observations_path)
if not os.path.isdir(f'{results_path}'): os.mkdir(results_path)

observations_file = f'{observations_path}/Observations.csv'
i = 1
while (os.path.isfile(f'{observations_file}')):
    observations_file = f'{observations_path}/Observations{i}.csv'
    i += 1
with open(observations_file, mode='w') as fp:
    pass

class DataManager:
    @staticmethod
    def store_observation(data):
        if not (isinstance(data, list) or isinstance(data, np.ndarray)):
            print("Invalid observation data type for function write\_data in PCA. Expected list or numpy.ndarray, got: " + str(type(data)))
            sys.exit()
        try:
            with open(observations_file, mode='a') as fp:
                data_writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                data_writer.writerow(data)
        except Exception as e:
            print("Writing data to csv failed.")
            print(e)
            sys.exit()

    @staticmethod
    def get_observations():
        pass

    @staticmethod
    def write_results(filename_base, rewards, steps, duration, setup):
        global results_path
        results_file = results_path + '/' + filename_base
        rows = zip(rewards, steps, duration)
        try:
            with open(f'{results_file}_results.csv', "w") as f:
                writer = csv.writer(f)
                writer.writerow(["Rewards", "steps", "Duration"])
                for row in rows:
                    writer.writerow(row)
            
            with open(f'{results_file}_setup.json', 'w') as f:
                json.dump(setup, f)
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

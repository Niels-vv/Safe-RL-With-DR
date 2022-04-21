from principal_component_analysis.pca_agent import train_pca
from utils.DataManager import DataManager
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("game_name", "MoveToBeacon", "PySC2 map name")

def get_pca_config(path):
    pca_config = {
        'file_path': path,
        'file_name' : 'pca.pt',
        'latent_space' : 16*16,
        'scalar' : True
    }
    return pca_config

def training_pca(unused_args):
    observation_sub_dir = f'../drive/MyDrive/Thesis/Code/PySC2/{FLAGS.game_name}/Observations'
    results_sub_dir = f'env_pysc2/results_pca/{FLAGS.game_name}'
    data_manager = DataManager(observation_sub_dir = observation_sub_dir, results_sub_dir = results_sub_dir)

    config = get_pca_config(results_sub_dir)
    get_statistics = True # If True, pca will first train a pca for all principal components, to get info about the variance on the used latent space

    train_pca(data_manager, config, get_statistics)

if __name__ == '__main__':
    app.run(training_pca)
from principal_component_analysis.pca_agent import train_pca
from utils.DataManager import DataManager
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "PongNoFrameskip-v4", "OpenAI name of the game/env whose obs to train on.")

def get_pca_config(path):
    pca_config = {
        'file_path': path,
        'file_name' : 'pca.pt',
        'latent_space' : 42*42,
        'scalar' : False
    }
    return pca_config

def training_pca(unused_args):
    observation_sub_dir = f'../drive/MyDrive/Thesis/Code/Atari/{FLAGS.game}/Observations'
    results_sub_dir = f'env_atari/results_pca/{FLAGS.game}'
    data_manager = DataManager(observation_sub_dir = observation_sub_dir, results_sub_dir = results_sub_dir)

    config = get_pca_config(results_sub_dir)

    train_pca(data_manager, config)

if __name__ == '__main__':
    app.run(training_pca)
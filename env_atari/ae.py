import torch
from env_atari.models import ae_encoder, ae_decoder
from autoencoder.ae_agent import train_ae
from utils.DataManager import DataManager
from absl import app, flags

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FLAGS = flags.FLAGS
flags.DEFINE_string("game", "PongNoFrameskip-v4", "OpenAI name of the game/env whose obs to train on.")

def get_ae_config(path):
    ae_config = {
        'lr' : 0.0001,
        'file_path': path,
        'file_name' : 'ae.pt',
        'batch_size' : 25,
        'latent_space' : 4*42*42
    }
    return ae_config

def training_ae(unused_args):
    observation_sub_dir = f'../drive/MyDrive/Thesis/Code/Atari/{FLAGS.game}/Observations'
    results_sub_dir = f'env_atari/results_ae/{FLAGS.game}'
    data_manager = DataManager(observation_sub_dir = observation_sub_dir, results_sub_dir = results_sub_dir)

    path = f'env_atari/results_ae/{FLAGS.game}'
    config = get_ae_config(path)
    train_ae(config, ae_encoder, ae_decoder, data_manager, device)

if __name__ == '__main__':
    app.run(training_ae)


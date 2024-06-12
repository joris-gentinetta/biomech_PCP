import argparse
import math
import os

import torch
from helpers.models import TorchTimeSeriesClassifier

import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import pandas as pd
from time import time
import wandb
from helpers.predict_utils import Config, train_model, TSDataset, TSDataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


side = 'Left'
channels = [0, 1, 2, 4, 5, 8, 10, 11]
features = [('emg', channel) for channel in channels]
targets = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'wristRot', 'wristFlex']
targets = [(side, target) for target in targets]

name = 'GRU'
model_path = f'model_files/{name}.pt'  # todo
config = Config({'features': features,
                         'targets': targets,
                         'n_epochs': 35,
                         'warmup_steps': 10,
                         'learning_rate': 0.01,
                         'hidden_size': 10,
                         'seq_len': 125,
                         'n_layers': 1,
                         'model_type': 'GRU'}
                        )
#
# name = 'GRU_big'
# config = Config({'features': features,
#                          'targets': targets,
#                          'n_epochs': 35,
#                          'warmup_steps': 10,
#                          'learning_rate': 0.001,
#                          'hidden_size': 500,
#                          'seq_len': 125,
#                          'n_layers': 1,
#                          'model_type': 'GRU'}
#                         )
#
# name = 'GRU_all_data'
# model_path = f'model_files/{name}.pt'  # todo
# config = Config({'features': features,
#                          'targets': targets,
#                          'n_epochs': 35,
#                          'warmup_steps': 10,
#                          'learning_rate': 0.01,
#                          'hidden_size': 10,
#                          'seq_len': 125,
#                          'n_layers': 1,
#                          'model_type': 'GRU'}
#                         )
#
# name = 'GRU_big_all_data'
# config = Config({'features': features,
#                          'targets': targets,
#                          'n_epochs': 35,
#                          'warmup_steps': 10,
#                          'learning_rate': 0.001,
#                          'hidden_size': 500,
#                          'seq_len': 125,
#                          'n_layers': 2,
#                          'model_type': 'GRU'}
#                         )



model = TorchTimeSeriesClassifier(input_size=len(config.features), hidden_size=config.hidden_size,
                                  output_size=len(config.targets), n_epochs=config.n_epochs,
                                  seq_len=config.seq_len,
                                  learning_rate=config.learning_rate,
                                  warmup_steps=config.warmup_steps, num_layers=config.n_layers,
                                  model_type=config.model_type)

model.load(model_path)
model.to(device)

if config.model_type == 'LSTM':
    states = (torch.zeros(config.n_layers, 1, config.hidden_size),
              torch.zeros(config.n_layers, 1, config.hidden_size))
else:
    states = torch.zeros(config.n_layers, 1, config.hidden_size)
output_dict = {target: 0 for target in targets}


##############################################   # todo
emg = np.load('/Users/jg/projects/biomech/DataGen/data/linda/minJerk/pinchCloseOpen/experiments/1/cropped_aligned_emg.npy')
for i in range(0, emg.shape[0]):
    emg_timestep = emg[i, channels]
##############################################


    emg_timestep = torch.tensor(emg_timestep, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output, states = model.model(emg_timestep, states)
        output = output.squeeze().to('cpu').detach().numpy()
        for j, target in enumerate(targets):
            output_dict[target] = output[j]
        output_dict['thumbInPlaneAng'] = output_dict['thumbInPlaneAng'] - math.pi
        output_dict['wristRot'] = (output_dict['wristRot'] * 2) - math.pi
        output_dict['wristFlex'] = (output_dict['wristFlex'] - math.pi / 2)
        # todo


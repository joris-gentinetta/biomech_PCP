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

parser = argparse.ArgumentParser(description='Timeseries data analysis')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
parser.add_argument('-t', '--test', action='store_true', help='Test the model')
parser.add_argument('-s', '--save_model', action='store_true', help='Save a model')
args = parser.parse_args()

# load the data:
angles_file = 'cropped_smooth_angles.parquet'
emg_file = 'cropped_aligned_emg.npy'
side = 'Left'
sampling_frequency = 60

channels = [0, 1, 2, 4, 5, 8, 10, 11]
recordings = [
    'fingersFlEx',
    'wristFlEx',
    'wristFlexHandOpen',
    'thumbFlEx',
    'handCloseOpen',
    'pinchCloseOpen',
    'indexFlEx',
    'indexFlexDigtisEx',
    'wristFlexHandClose',
    'digitsFlEx'
]

features = [('emg', channel) for channel in channels]
targets = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'wristRot', 'wristFlex']
targets = [(side, target) for target in targets]
trainsets = []
testsets = []
combined_sets = []
for recording in recordings:
    data_dir = f'/Users/jg/projects/biomech/DataGen/data/linda/minJerk/{recording}/experiments/1'
    angles = pd.read_parquet(f'{data_dir}/{angles_file}')
    angles.index = range(len(angles))
    emg = np.load(f'{data_dir}/{emg_file}')

    data = angles[targets].copy()
    data.loc[:, (side, 'thumbInPlaneAng')] = data.loc[:, (side, 'thumbInPlaneAng')] + math.pi
    data.loc[:, (side, 'wristRot')] = (data.loc[:, (side, 'wristRot')] + math.pi) / 2
    data.loc[:, (side, 'wristFlex')] = (data.loc[:, (side, 'wristFlex')] + math.pi / 2)

    data = data / math.pi
    data = np.clip(data, 0, 1)

    for channel in channels:
        data[('emg', channel)] = emg[:, channel]

    # plot all features:
    if args.visualize:
        emg_columns = [col for col in data.columns if col[0] == 'emg']
        data[emg_columns].plot(subplots=True)
        plt.show()

        data[targets].plot(subplots=True)
        plt.show()

    test_set = data.loc[len(data) // 5 * 4:].copy()
    # test_angles = angles.iloc[len(data) // 5 * 4:].copy()
    train = data.loc[: len(data) // 5 * 4].copy()
    trainsets.append(train)
    testsets.append(test_set)
    combined_sets.append(data.copy())

# number of steps to warm up the model before evaluating the loss function:
warmup_steps = 10

if args.hyperparameter_search:
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'mean_best_val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'features': {
                'value': features
            },
            'targets': {
                'value': targets
            },
            'n_epochs': {
                'value': 2500
            },
            'warmup_steps': {
                'value': 10
            },
            'learning_rate': {
                'values': [0.01, 0.001]
            },
            'hidden_size': {
                'values': [10, 20]
            },
            'seq_len': {
                'value': 125
            },
            'n_layers': {
                'values': [1, 2]
            },
            'model_type': {
                'values': ['CNN', 'RNN', 'LSTM', 'GRU']  # ['CNN', 'RNN', 'LSTM', 'GRU']
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="bm_10")
    wandb.agent(sweep_id, lambda config=None: train_model(trainsets, testsets, config=config))

if args.test:
    cnn_config = Config({'features': features,
                         'targets': targets,
                         'n_epochs': 55,
                         'warmup_steps': 10,
                         'learning_rate': 0.01,
                         'hidden_size': 10,
                         'seq_len': 125,
                         'n_layers': 2,
                         'model_type': 'GRU'}
                        )
    configs = {'CNN': cnn_config}
    for name, config in configs.items():
        device = torch.device("cpu")
        model = TorchTimeSeriesClassifier(input_size=len(config.features), hidden_size=config.hidden_size,
                                          output_size=len(config.targets), n_epochs=config.n_epochs,
                                          seq_len=config.seq_len,
                                          learning_rate=config.learning_rate,
                                          warmup_steps=config.warmup_steps, num_layers=config.n_layers,
                                          model_type=config.model_type)
        model.to(device)
        dataset = TSDataset(trainsets, config.features, config.targets, sequence_len=125)
        dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

        for epoch in range(model.n_epochs):
            model.train_one_epoch(dataloader)

        for set_id, test_set in enumerate(testsets):
            val_pred = model.predict(test_set, config.features).squeeze(0).to(device).detach().numpy()

            val_pred = np.clip(val_pred, 0, 1) * math.pi
            test_set[config.targets] = val_pred

            test_set.loc[:, (side, 'thumbInPlaneAng')] = test_set.loc[:, (side, 'thumbInPlaneAng')] - math.pi
            test_set.loc[:, (side, 'wristRot')] = (test_set.loc[:, (side, 'wristRot')] * 2) - math.pi
            test_set.loc[:, (side, 'wristFlex')] = (test_set.loc[:, (side, 'wristFlex')] - math.pi / 2)
            data_dir = f'/Users/jg/projects/biomech/DataGen/data/linda/minJerk/{recordings[set_id]}/experiments/1'
            test_set.to_parquet(join(data_dir, f'pred_angles_{name}.parquet'))

if args.save_model:
    gru_config = Config({'features': features,
                         'targets': targets,
                         'n_epochs': 35,
                         'warmup_steps': 10,
                         'learning_rate': 0.01,
                         'hidden_size': 10,
                         'seq_len': 125,
                         'n_layers': 1,
                         'model_type': 'GRU'}
                        )
    configs = {'GRU': gru_config}
    for name, config in configs.items():
        device = torch.device("cpu")
        model = TorchTimeSeriesClassifier(input_size=len(config.features), hidden_size=config.hidden_size,
                                          output_size=len(config.targets), n_epochs=config.n_epochs,
                                          seq_len=config.seq_len,
                                          learning_rate=config.learning_rate,
                                          warmup_steps=config.warmup_steps, num_layers=config.n_layers,
                                          model_type=config.model_type)
        model.to(device)
        dataset = TSDataset(combined_sets, config.features, config.targets, sequence_len=125)
        dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

        train_losses = []
        for epoch in range(model.n_epochs):
            loss = model.train_one_epoch(dataloader)
            train_losses.append(loss)

        plt.plot(train_losses)
        plt.show()

        model.to(torch.device('cpu'))
        os.makedirs(f'model_files', exist_ok=True)
        model.save(f'model_files/{name}.pt')

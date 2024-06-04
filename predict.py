import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
import wandb
from helpers.predict_utils import Config, train_model

parser = argparse.ArgumentParser(description='Timeseries data analysis')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
parser.add_argument('-t', '--test', action='store_true', help='Test the model')
args = parser.parse_args()

# load the data:
angles_file = 'smooth_angles.parquet'
emg_file = 'aligned_emg.npy'
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
for recording in recordings:
    data_dir = f'/Users/jg/projects/biomech/DataGen/data/linda/minJerk/{recording}/experiments/1'
    angles = pd.read_parquet(f'{data_dir}/{angles_file}')
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

    test_set = data.iloc[len(data) // 5 * 4:].copy()
    test_angles = angles.iloc[len(data) // 5 * 4:].copy()
    train = data.iloc[: len(data) // 5 * 4].copy()
    trainsets.append(train)
    testsets.append(test_set)

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
                'values': [0.1, 0.01, 0.001, 0.0001]
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
                         'n_epochs': 2,
                         'warmup_steps': 10,
                         'learning_rate': 0.005,
                         'hidden_size': 10,
                         'seq_len': 125,
                         'n_layers': 2,
                         'model_type': 'GRU'}
                        )
    configs = {'CNN': cnn_config}
    for name, config in configs.items():
        train_model(trainsets, testsets, mode='disabled', config=config)
        # for test_set in testsets:
        #     pred = model.predict(test_set, features)
        #     pred = pred.detach().numpy().squeeze()
        #     test_angles.loc[:, targets] = np.clip(pred, 0, 1) * math.pi
        #     data.loc[:, (side, 'thumbInPlaneAng')] = data.loc[:, (side, 'thumbInPlaneAng')] - math.pi
        #     data.loc[:, (side, 'wristRot')] = (data.loc[:, (side, 'wristRot')] * 2) - math.pi
        #     data.loc[:, (side, 'wristFlex')] = (data.loc[:, (side, 'wristFlex')] - math.pi / 2)
            # test_angles.to_parquet(join(data_dir, f'test_angles_{name}.parquet'))

        # preds.append(pred[warmup_steps:])
        # targets.append(test_set['y'].values[warmup_steps:])
        #
        # pred = np.concatenate(preds)
        # target = np.concatenate(targets)
        # binary_pred = (pred > 0.5).astype(int)
        #
        # accuracy = (binary_pred == target).mean()
        # sensitivity = ((binary_pred == 1) & (target == 1)).sum() / (target == 1).sum()
        # specificity = ((binary_pred == 0) & (target == 0)).sum() / (target == 0).sum()
        # auc = roc_auc_score(target, pred)
        # plot_auc_roc(target, pred, name)
        # plot_results(target, pred, binary_pred, name)
        #
        #
        # print('\n#############################################')
        # print(f'{name} accuracy: {accuracy}')
        # print(f'{name} sensitivity: {sensitivity}')
        # print(f'{name} specificity: {specificity}')
        # print(f'{name} auc: {auc}')

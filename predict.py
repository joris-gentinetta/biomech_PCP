import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from helpers.predict_utils import plot_auc_roc, Config, plot_results, train_model, TSDataset, TSDataLoader
from helpers.models import TorchTimeSeriesClassifier
import wandb
import math
from os.path import join


parser = argparse.ArgumentParser(description='Timeseries data analysis')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
parser.add_argument('-d', '--debug', action='store_true', help='Use debug mode')
parser.add_argument('-t', '--test', action='store_true', help='Test the model')
args = parser.parse_args()


# load the data:
data_dir = '/Users/jg/projects/biomech/DataGen/data/joris/trigger_1/experiments/1'
angles_file = 'smooth_angles.parquet'
emg_file = 'aligned_emg.npy'
sampling_frequency = 60  # assuming the data is sampled at 60 Hz
angles = pd.read_parquet(f'{data_dir}/{angles_file}')
emg = np.load(f'{data_dir}/{emg_file}')

side = 'Left'
channels = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
features = [('emg', channel) for channel in channels]
targets = ['indexAng', 'midAng', 'ringAng', 'pinkyAng', 'thumbInPlaneAng', 'thumbOutPlaneAng', 'elbowAngle', 'wristRot', 'wristFlex']
targets = [(side, target) for target in targets]

# create a dataframe containing Left/idexAng from angles as y the used channels of emg as x_[channel]:
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


# take first 1/10 and last 1/10 of data for testing:
test_set = data.iloc[len(data)//5 * 4 :].copy()
test_angles = angles.iloc[len(data)//5 * 4 :].copy()
# take the rest for training:
train = data.iloc[: len(data)//5 * 4].copy()

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
            'max_n_epochs': {
                'value': 2500
            },
            'warmup_steps': {
                'value': 10
            },
            'used_features': {
                'value': features
            },
            'learning_rate': {
                'values': [0.0005, 0.005, 0.01, 0.5]
            },
            'hidden_size': {
                'values': [10]
            },
            'seq_len': {
                'value': 125
            },
            'n_layers': {
                'values': [1]
            },
            'model_type': {
                'values': ['CNN', 'RNN', 'LSTM', 'GRU']
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="sovn_5")
    wandb.agent(sweep_id, lambda config=None: train_model([folds[0]], config))


if args.test:
    cnn_config = Config({'n_epochs': 100, 'learning_rate': 0.01, 'hidden_size': 10, 'seq_len': 125, 'n_layers': 1, 'model_type': 'CNN'})
    gru_config = Config({'n_epochs': 100, 'learning_rate': 0.0005, 'hidden_size': 10, 'seq_len': 125, 'n_layers': 2, 'model_type': 'GRU'})
    configs = {'CNN': cnn_config, 'GRU': gru_config}
    for name, config in configs.items():
        model = TorchTimeSeriesClassifier(input_size=len(features), hidden_size=config.hidden_size, output_size=len(targets), n_epochs=config.n_epochs, seq_len=config.seq_len, learning_rate=config.learning_rate,
                                                  warmup_steps=warmup_steps, num_layers=config.n_layers, model_type=config.model_type)
        dataset = TSDataset(train, features, targets, sequence_len=125)
        dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True)
        for epoch in range(model.n_epochs):
            model.train_one_epoch(dataloader)

        pred = model.predict(test_set, features)
        pred = pred.detach().numpy().squeeze()
        test_angles.loc[:, targets] = np.clip(pred, 0, 1) * math.pi
        data.loc[:, (side, 'thumbInPlaneAng')] = data.loc[:, (side, 'thumbInPlaneAng')] - math.pi
        data.loc[:, (side, 'wristRot')] = (data.loc[:, (side, 'wristRot')] * 2) - math.pi
        data.loc[:, (side, 'wristFlex')] = (data.loc[:, (side, 'wristFlex')] - math.pi / 2)
        test_angles.to_parquet(join(data_dir, f'test_angles_{name}.parquet'))



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







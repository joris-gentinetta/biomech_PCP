import argparse
import math
import os
import yaml
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from helpers.models import TimeSeriesRegressorWrapper
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import pandas as pd
import wandb
from tqdm import tqdm
from helpers.predict_utils import Config, train_model, TSDataset, TSDataLoader


parser = argparse.ArgumentParser(description='Timeseries data analysis')
parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
parser.add_argument('-t', '--test', action='store_true', help='Test the model')
parser.add_argument('-s', '--save_model', action='store_true', help='Save a model')
args = parser.parse_args()

sampling_frequency = 60

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('Using MPS')
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using CUDA')
else:
    device = torch.device("cpu")
    print('Using CPU')


with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
    wandb_config = yaml.safe_load(file)
    config = Config(wandb_config)
data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in config.recordings]


trainsets = []
testsets = []
combined_sets = []
for data_dir in data_dirs:
    angles = pd.read_parquet(join(data_dir, 'cropped_smooth_angles.parquet'))
    angles.index = range(len(angles))
    emg = np.load(join(data_dir, 'cropped_aligned_emg.npy'))

    data = angles.copy()
    data.loc[:, (args.intact_hand, 'thumbInPlaneAng')] = data.loc[:, (args.intact_hand, 'thumbInPlaneAng')] + math.pi
    data.loc[:, (args.intact_hand, 'wristRot')] = (data.loc[:, (args.intact_hand, 'wristRot')] + math.pi) / 2
    data.loc[:, (args.intact_hand, 'wristFlex')] = (data.loc[:, (args.intact_hand, 'wristFlex')] + math.pi / 2)

    data = data / math.pi
    data = np.clip(data, 0, 1)

    for feature in config.features:
        data[feature] = emg[:, feature[1]]

    if args.visualize:
        data[config.features].plot(subplots=True)
        plt.title('Features')
        plt.show()

        data[config.targets].plot(subplots=True)
        plt.title('Targets')
        plt.show()

    test_set = data.loc[len(data) // 5 * 4:].copy()
    train = data.loc[: len(data) // 5 * 4].copy()
    trainsets.append(train)
    testsets.append(test_set)
    combined_sets.append(data.copy())


if args.hyperparameter_search:
    sweep_id = wandb.sweep(wandb_config, project=f'{args.person_dir}_{args.config_name}')
    wandb.agent(sweep_id, lambda config=None: train_model(trainsets, testsets, config=config))


if args.test:
    model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets), **(config.to_dict()))

    model.to(device)
    dataset = TSDataset(trainsets, config.features, config.targets, sequence_len=125, device=device)
    dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    print('Training model...')
    for epoch in tqdm(range(model.n_epochs)):
        model.train_one_epoch(dataloader)

    for set_id, test_set in enumerate(testsets):
        val_pred = model.predict(test_set, config.features).squeeze(0).to('cpu').detach().numpy()

        val_pred = np.clip(val_pred, 0, 1) * math.pi
        test_set[config.targets] = val_pred

        test_set.loc[:, (args.intact_hand, 'thumbInPlaneAng')] = test_set.loc[:, (args.intact_hand, 'thumbInPlaneAng')] - math.pi
        test_set.loc[:, (args.intact_hand, 'wristRot')] = (test_set.loc[:, (args.intact_hand, 'wristRot')] * 2) - math.pi
        test_set.loc[:, (args.intact_hand, 'wristFlex')] = (test_set.loc[:, (args.intact_hand, 'wristFlex')] - math.pi / 2)
        test_set.to_parquet(join(data_dirs[set_id], f'pred_angles-{args.config_name}.parquet'))


if args.save_model:
    model = TimeSeriesRegressorWrapper(input_size=len(config.features), hidden_size=config.hidden_size,
                                      output_size=len(config.targets), n_epochs=config.n_epochs,
                                      seq_len=config.seq_len,
                                      learning_rate=config.learning_rate,
                                      warmup_steps=config.warmup_steps, num_layers=config.n_layers,
                                      model_type=config.model_type)
    model.to(device)
    dataset = TSDataset(combined_sets, config.features, config.targets, sequence_len=125, device=device)
    dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    print('Training model...')
    for epoch in tqdm(range(model.n_epochs)):
        model.train_one_epoch(dataloader)

    model.to(torch.device('cpu'))
    os.makedirs(join('data', args.person_dir, 'models'), exist_ok=True)
    model.save(join('data', args.person_dir, 'models', f'{args.config_name}.pt'))

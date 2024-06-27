
import argparse
import math
import os
import yaml
import gc

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch

from os.path import join
import pandas as pd
import wandb
from tqdm import tqdm
import numpy as np

import torch.optim as optim
from helpers.predict_utils import Config, train_model, TSDataset, TSDataLoader
from helpers.UpperTrainer import Offline_Trainer
from Dynamics2.upperLimbModel_copy import upperExtremityModel


np.set_printoptions(linewidth=200, suppress=True)
torch.set_printoptions(linewidth=300, sci_mode=False)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print('Using MPS')
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using CUDA')
else:
    device = torch.device("cpu")
    print('Using CPU')

parser = argparse.ArgumentParser(description='Timeseries data analysis')
parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
parser.add_argument('-t', '--test', action='store_true', help='Test the model')
parser.add_argument('-s', '--save_model', action='store_true', help='Save a model')
args = parser.parse_args()


EMG_mapping_Lr = 10
Dynamic_system_Lr = 5
Lr = 1e-2
NN_ratio = 0.2

sampling_frequency = 60
muscleType = 'bilinear'


if __name__ == '__main__':

    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)
    data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.recordings]

    trainsets = []
    testsets = []
    combined_sets = []
    for data_dir in data_dirs:
        angles = pd.read_parquet(join(data_dir, 'cropped_smooth_angles.parquet'))
        angles.index = range(len(angles))
        emg = np.load(join(data_dir, 'cropped_aligned_emg.npy'))

        data = angles.copy()
        data.loc[:, (args.intact_hand, 'thumbInPlaneAng')] = data.loc[:,
                                                             (args.intact_hand, 'thumbInPlaneAng')] + math.pi
        data.loc[:, (args.intact_hand, 'wristRot')] = (data.loc[:, (args.intact_hand, 'wristRot')] + math.pi) / 2
        data.loc[:, (args.intact_hand, 'wristFlex')] = (data.loc[:, (args.intact_hand, 'wristFlex')] + math.pi / 2)

        data = data / math.pi
        data = np.clip(data, 0, 1)

        for feature in config.features:
            data[feature] = emg[:, feature[1]]


        test_set = data.loc[len(data) // 5 * 4:].copy()
        train = data.loc[: len(data) // 5 * 4].copy()
        trainsets.append(train)
        testsets.append(test_set)
        combined_sets.append(data.copy())

    train_dataset = TSDataset(trainsets, config.features, config.targets, sequence_len=config.seq_len, dummy_labels=True, device=device)
    train_dataloader = TSDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    test_dataset = TSDataset(testsets, config.features, config.targets, sequence_len=config.seq_len, dummy_labels=True, device=device)
    test_dataloader = TSDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True)
    data_loaders = {'train': train_dataloader, 'test': test_dataloader}
    a = len(train_dataloader)

    torch.cuda.empty_cache()
    gc.collect()


    print(f'Using {torch.get_num_threads()} CPU threads')
    system_dynamic_model = upperExtremityModel(muscleType=muscleType, numDoF=len(config.targets), device=device, EMG_Channel_Count=len(config.features), Dynamic_Lr=Dynamic_system_Lr, EMG_mat_Lr=EMG_mapping_Lr, NN_ratio=NN_ratio)

    print(f'Number of model parameters: {system_dynamic_model.count_params()}')

    model_save_path = "./minJerkModel_synergyNoCost.tar"

    trainer = Offline_Trainer(system_dynamic_model, model_save_path, device, early_stopping=3, warmup_steps=config.warmup_steps, clip=10, EMG_mapping_flexible=EMG_mapping_Lr)
    optimizer = optim.Adam(system_dynamic_model.parameters(), lr=Lr, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, cooldown=0)




    trainer.train(data_loaders, optimizer, scheduler, init_model_path=None, epoch_num=config.n_epochs, twoStep=True)

    system_dynamic_model.print_params()

    torch.save({'model_state_dict': system_dynamic_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                model_save_path)
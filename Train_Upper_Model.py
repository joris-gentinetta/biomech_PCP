
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
from helpers.predict_utils import Config, train_model, TSDataset, TSDataLoader
import numpy as np
np.set_printoptions(linewidth=200, suppress=True)
torch.set_printoptions(linewidth=300, sci_mode=False)


import sys
import os

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
stride = 1
batch_size = 512

INPUT_FS = 1000
OUTPUT_FS = 60

muscleType = 'bilinear'



# Add to the path this folder and its parent to find the other modules
sys.path.append(os.path.dirname(os.path.abspath('')))

if __name__ == '__main__':
    #%% md
    # # Load Dataset from Pickle files
    #%%
    # FOLDER_PATH = '/home/haptix/UE AMI Clinical Work/Patient Data/P7 - 453/P7_0325_2024/Data Recording/PROCESSED/pkl/'
    FOLDER_PATH = '/Users/jg/projects/biomech/DataGen/biomech_Shadmehr_Opt/data/P7_0326_2024/MinJerk/'
    FILE_NAMES = []
    for path in sorted(os.listdir(FOLDER_PATH)):
        if path.endswith('.pkl') and os.path.isfile(os.path.join(FOLDER_PATH, path)):
            FILE_NAMES.append(path) # this should end in .pkl

    print(FILE_NAMES)
    #%%
    def jointAdjustment(jointNames, jointAngles, splits, scaling=1, handedness='L'):
        if handedness == 'R':
            jointRoMs = {'thumbYPos': [0, 75], 'thumbPPos': [0, 100], 'indexPos': [0, 90], 'mrpPos': [0, 90], 'wristRot': [-120, 175], 'wristFlex': [-55, 55], 'humPos': [-95, 95], 'elbowPos': [0, 135]}
        elif handedness == 'L':
            jointRoMs = {'thumbYPos': [0, 75], 'thumbPPos': [0, 100], 'indexPos': [0, 90], 'mrpPos': [0, 90], 'wristRot': [-175, 120], 'wristFlex': [-55, 55], 'humPos': [-95, 95], 'elbowPos': [0, 135]}
        else:
            raise ValueError(f'Invalid handedness {handedness}')

        numJoints = len(jointNames)
        assert numJoints == jointAngles.shape[1], f'Different number of joints names and angles given ({numJoints} names and {jointAngles.shape[1]} joints)'

        returnAnglesRaw = np.zeros_like(jointAngles)
        for joint in range(numJoints):
            angleRaw = jointAngles[:, joint]
            jointRoM = jointRoMs[jointNames[joint]]
            rom = np.deg2rad(sum([abs(i) for i in jointRoM]))

            returnAnglesRaw[:, joint] = angleRaw*scaling/rom

        returnAngles = [returnAnglesRaw[splits[i]:splits[i + 1], :] for i in range(len(splits) - 1)]
        returnAngles = [[np.array(j) for j in i] for i in returnAngles]

        return returnAngles


    sampling_frequency = 60

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

    train_dataset = TSDataset(trainsets, config.features, config.targets, sequence_len=125, dummy_labels=True, device=device)
    train_dataloader = TSDataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_dataset = TSDataset(testsets, config.features, config.targets, sequence_len=125, dummy_labels=True, device=device)
    test_dataloader = TSDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    data_loaders = {'train': train_dataloader, 'test': test_dataloader}
    a = len(train_dataloader)
######################################################################

    torch.cuda.empty_cache()
    gc.collect()
    #%%

    # from Dynamics2.upperLimbModel import upperExtremityModel # use the new version of the model, the generic one
    from Dynamics2.upperLimbModel_copy import upperExtremityModel

    print(f'Using {torch.get_num_threads()} CPU threads')
    system_dynamic_model = upperExtremityModel(muscleType=muscleType, numDoF=len(config.targets), device=device, EMG_Channel_Count=len(config.features), Dynamic_Lr=Dynamic_system_Lr, EMG_mat_Lr=EMG_mapping_Lr, NN_ratio=NN_ratio)

    print(f'Number of model parameters: {system_dynamic_model.count_params()}')


    #%%
    import torch.optim as optim
    # from Upper_Trainer import Offline_Trainer
    from helpers.UpperTrainer import Offline_Trainer

    model_save_path = "./minJerkModel_synergyNoCost.tar"

    system_dynamic_model.train() # put the model in train mode
    trainer = Offline_Trainer(system_dynamic_model, model_save_path, device, early_stopping=3, warmup_steps=10, clip=10, EMG_mapping_flexible=EMG_mapping_Lr)
    optimizer = optim.Adam(system_dynamic_model.parameters(), lr=Lr, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, cooldown=0)
    #%%
    # trainer.train(data_loaders, optimizer, scheduler, init_model_path='minJerkModel_synergyNoCost.tar', epoch_num=1000, twoStep=True)
    trainer.train(data_loaders, optimizer, scheduler, init_model_path=None, epoch_num=1000, twoStep=True)

    # trainer.train(data_loaders=data_loaders, optimizer=optimizer, scheduler=scheduler, epoch_num=1000, twoStep=True)
    #%%
    # for name, param in system_dynamic_model.named_parameters():
    #     if param.requires_grad and not (name.__contains__('compensational_nns') or name.__contains__('recognitionLayer') or name.__contains__('activationLayer')):
    #         print(name, param.data)
    system_dynamic_model.print_params()

    #%%
    torch.save({'model_state_dict': system_dynamic_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                model_save_path)
#%% md
# # Train the controller with minimum jerk assumption
# 
# BUGMAN Feb 17 2022
#%%
import argparse
import math
import os
import yaml

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from helpers.models import TorchTimeSeriesClassifier
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import pandas as pd
import wandb
from tqdm import tqdm
from helpers.predict_utils import Config, train_model, TSDataset, TSDataLoader
import numpy as np
np.set_printoptions(linewidth=200, suppress=True)
torch.set_printoptions(linewidth=300, sci_mode=False)

from biomech_Shadmehr_Opt.OnlineLearning.Offline_EMG_Goniometer_DataLoader import Offline_EMG_Goniometer_DataLoader


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

EMG_Channel_Count = 8


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

    #%%

    #%%################################################################
    # from biomech_Shadmehr_Opt.OnlineLearning.Offline_Dataset import Offline_Dataset
    #
    # data_set = Offline_Dataset(processed_EMG_list, processed_Angle_list, processed_Joint_list, data_piece_length=120, stride=stride)
    # numTrainingPoints = len(data_set)
    # print(f'Number of data points: {numTrainingPoints}')
    #
    # trainCount = int(0.8*numTrainingPoints)
    # testCount = numTrainingPoints - trainCount
    # train_dataset, test_dataset = torch.utils.data.random_split(data_set, (trainCount, testCount))
    #
    # trainDataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True, pin_memory=True)
    # testDataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, persistent_workers=False, drop_last=True, pin_memory=True)
    #
    # data_loaders = {'train': trainDataLoader,
    #               'test': testDataLoader}
    #
    # print(f'Train batches: {len(trainDataLoader)}')
    # print(f'Test batches: {len(testDataLoader)}')
    #%%################################################################

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
    dataloader = TSDataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_dataset = TSDataset(testsets, config.features, config.targets, sequence_len=125, dummy_labels=True, device=device)
    test_dataloader = TSDataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    data_loaders = {'train': dataloader, 'test': test_dataloader}

######################################################################


    #%%
    # checking the dataloader
    # for i in dataloader:
    #     print()
    #%%
    print()
    #%% md
    # # Train the Model
    #%%
    import gc

    torch.cuda.empty_cache()
    gc.collect()
    #%%

    # from Dynamics2.upperLimbModel import upperExtremityModel # use the new version of the model, the generic one
    from Dynamics2.upperLimbModel_copy import upperExtremityModel

    print(f'Using {torch.get_num_threads()} CPU threads')
    system_dynamic_model = upperExtremityModel(muscleType=muscleType, numDoF=len(config.targets), device=device, EMG_Channel_Count=EMG_Channel_Count, Dynamic_Lr=Dynamic_system_Lr, EMG_mat_Lr=EMG_mapping_Lr, NN_ratio=NN_ratio)

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
    #%% md
    # # Verify the model
    #%%
    # This block is only to reload things
    import pickle

    FILE_NAMES = ['P1_0307_2022_elbow.pkl', 'P1_0307_2022_finger.pkl', 'P1_0307_2022_thumb.pkl', 'P1_0307_2022_index.pkl']

    file_data_loader = Offline_EMG_Goniometer_DataLoader()

    processed_EMG_list = []
    processed_Angle_list = []

    for FILE_NAME in FILE_NAMES:
        print(FILE_NAME)
        # FOLDER_PATH = '/home/haptix/haptix/Upper Extremity Models/Upper Extremity Shadmehr/' + FILE_NAME
        FOLDER_PATH = '/home/haptix/UE AMI Clinical Work/P1 - 729/P1_0307_2022/MinJerk Pickle/' + FILE_NAME

        with open(FOLDER_PATH, "rb") as f:
            loaded_data = pickle.load(f)

        processed_EMG_list.append(loaded_data[0])
        processed_Angle_list.append(loaded_data[1])
    #%%
    model_save_path = "./minJerkModel_synergyNoCost.tar"
    # model_save_path = '/home/haptix/UE AMI Clinical Work/P3 - 152/P3_0726_2022/Models/5DoF_model_lambda2.tar'
    if True:
        system_dynamic_model.to(device)
        checkpoint = torch.load(model_save_path)
        # print(checkpoint['model_state_dict'])
        system_dynamic_model.load_state_dict(checkpoint['model_state_dict'])
    system_dynamic_model.eval()
    #%%
    ss = torch.tensor([[0]*numDoF*system_dynamic_model.numStates], dtype=torch.float, device=device)

    all_outs = [[] for _ in range(numDoF)]
    preds = [[] for _ in range(numDoF)]
    alphas = [[] for _ in range(numDoF)]

    with torch.no_grad():
        # Run the model simulation
        for e in input_EMG:
            input = torch.FloatTensor(np.array([e])).to(device)
            w, ss, probs, activation = system_dynamic_model(ss, input, dt=0.0166667)

            for i in range(numDoF):
                preds[i].append(probs[:, i].detach().cpu().numpy())
                all_outs[i].append((w[:, i]).detach().cpu().numpy())
                alphas[i].append(activation[:, i].detach().cpu().numpy())

        for i in range(numDoF):
            all_outs[i] = np.array(all_outs[i]).T[0]
            preds[i] = np.array(preds[i]).T[0]
            alphas[i] = np.array(alphas[i]).T[0]
    #%%
    # Save the model output
    import pandas as pd
    modelType = 'noNN_equalRoM_residual_longer2'
    jointLabels = np.concatenate(processed_Joint_list)
    dataToSave = np.hstack((plot_Angle, np.asarray(all_outs + preds).T, jointLabels))
    colNames = ['ref' + str(i) for i in range(numDoF)] + ['pos' + str(i) for i in range(numDoF)] + ['pred' + str(i) for i in range(numDoF)] + ['label' + str(i) for i in range(numDoF)]

    df = pd.DataFrame(data=dataToSave, columns=colNames)
    # df.to_csv('5DoFModelResults_' + modelType + '.csv', index=False)
    #%%

    t_goni = np.array(np.linspace(0, len(all_outs[0]), len(all_outs[0])))/60
    fig, axs = plt.subplots(numDoF, squeeze=False)
    fig.suptitle('Trained Model Performance')

    for i in range(numDoF):
        mus = FILE_NAMES[i]
        axs[i][0].plot(t_goni, all_outs[i], label=f'Model{i}')
        axs[i][0].plot(t_goni, np.array(plot_Angle)[:, i], label=f'MinJerk{i}')
        axs[i][0].plot(t_goni, preds[i], label=f'Pred{i}')
        axs[i][0].set_ylabel("Angle (rad)")

    axs[numDoF - 1][0].set_xlabel("Time (s)")
    # axs[numDoF - 1][0].legend()

    plt.show()
    #%%
    # Plots and calculations

    t_goni = np.array(np.linspace(0, len(all_outs[0]), len(all_outs[0])))/60
    fig, axs = plt.subplots(numDoF)
    # plt.figure(0)
    # plt.title("Trained Model Performance")
    fig.suptitle('Trained Model Performance', fontsize=20, y=0.95, x=0.5)
    # fig.suptitle('Model Output in Blue; Reference Trajectory in Orange', fontsize=16)

    plotAngles = np.array(plot_Angle)
    mse = (np.square(all_outs - plotAngles.T)).mean(axis=1)
    outs = (np.array(all_outs).T - np.array(all_outs).T.mean(axis=0)) / np.array(all_outs).T.std(axis=0)
    angs = (plotAngles - plotAngles.mean(axis=0)) / plotAngles.std(axis=0)
    pearson_r = np.dot(outs.T, angs) / outs.shape[0]

    residuals = np.sum(np.square(all_outs - plotAngles.T), axis=1)
    tss = np.sum(np.square(plotAngles.T - np.mean(plotAngles.T, axis=1)[:, None]), axis=1)
    R = 1 - residuals/tss
    print(f'R: {R}')

    joints =  ['Thumb', 'Index', 'Digits', 'Wrist rotate', 'Wrist flex']
    for i in range(numDoF):
        mus = FILE_NAMES[i]
        # plt.plot(t_goni, all_outs[i], label=f'Model Output {i}')
        # plt.plot(t_goni, np.array(plot_Angle)[:, i], label=f'Minimum Jerk {i}')
        axs[i].plot(t_goni, all_outs[i], label=f'Model Output {i}', linewidth=2.5)
        axs[i].plot(t_goni, np.array(plot_Angle)[:, i], label=f'Minimum Jerk {i}', linewidth=2)
        axs[i].set_ylabel(joints[i], fontsize=12)
        axs[i].autoscale(enable=True, axis='x', tight=True)

        # plt.plot(t_goni, preds[i], label=f'Predictions {i}')
        # plt.plot(t_goni, all_outs[i], label=f'Model Output {str(i)}')
        # plt.plot(t_goni, np.array(plot_Angle)[:, i], label=f'Minimum Jerk {str(i)}')

    axs[numDoF - 1].set_xlabel('Time (s)', fontsize=12)
    axs[0].set_title('Model Output in Blue; Reference Trajectory in Orange', fontsize=16)
    print(f'Before mse: {mse}')
    print(f'Before r:\n{pearson_r}')
    # plt.tight_layout()

    plt.show()

    fig2, axs2 = plt.subplots(numDoF)
    fig2.suptitle('Prediction of Intent to Move', fontsize=20, y=0.95, x=0.5)

    jointLabels = np.concatenate(processed_Joint_list)
    # cutoffs = np.array([0.6, 0.7, 0.7, 0.45, 0.45]) # JM 09-28, 09-28
    # cutoffs = np.array([0.5, 0.5, 0.3, 0.5]) # JM 09-28, 09-28
    # cutoffs = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    cutoffs = np.array([0.7, 0.7, 0.7, 0.7])

    joints =  ['ThumbP', 'ThumbY', 'Index', 'Digits', 'Wrist flex']
    smoothedArr = np.zeros((numDoF, len(preds[0])))
    for i in range(numDoF):
        x = 10
        smoothedArr[i, :] = np.convolve(preds[i], np.ones(x), 'same')/x
        # axs2[i].plot(t_goni, preds[i], 'b--', label=f'Prediction {i}', linewidth=1)
        axs2[i].plot(t_goni, smoothedArr[i, :], 'b--', label=f'Prediction {i}', linewidth=1.5)
        # axs2[i].plot(t_goni, cutoffs[i]*np.ones_like(preds[1]), 'r-.', label=f'Prediction {i}', linewidth=1)
        axs2[i].plot(t_goni, smoothedArr[i, :] > cutoffs[i], 'darkgreen', label=f'Prediction {i}', linewidth=2.5)
        axs2[i].plot(t_goni, jointLabels[:, i], label=f'Prediction {i}', linewidth=1.5)
        axs2[i].set_ylabel(joints[i], fontsize=12)
        axs2[i].autoscale(enable=True, axis='x', tight=True)

    axs2[numDoF - 1].set_xlabel('Time (s)', fontsize=12)
    axs2[0].set_title('Predictions in Blue, Cutoffs in Red, Thresholded Predictions in Green', fontsize=16)

    # plt.tight_layout()

    plt.show()
    newOuts = np.where(smoothedArr > np.repeat(cutoffs[:, None], repeats=smoothedArr.shape[1], axis=1), np.array(all_outs), 0)
    mse = (np.square(newOuts - plotAngles.T)).mean(axis=1)
    outs = (np.array(newOuts).T - np.array(newOuts).T.mean(axis=0)) / np.array(newOuts).T.std(axis=0)
    angs = (plotAngles - plotAngles.mean(axis=0)) / plotAngles.std(axis=0)
    pearson_r = np.dot(outs.T, angs) / outs.shape[0]

    print(f'After mse: {mse}')
    print(f'After r:\n{pearson_r}')
    jointLabels = np.concatenate(processed_Joint_list)
    correctLabel = np.logical_and(jointLabels.astype(bool), newOuts.T.astype(bool))
    numRight = np.count_nonzero(correctLabel, axis=0)
    numTrues = np.count_nonzero(jointLabels, axis=0)
    print(f'Classification accuracy:\n{np.divide(numRight, numTrues)}')

    residuals = np.sum(np.square(newOuts - plotAngles.T), axis=1)
    tss = np.sum(np.square(plotAngles.T - np.mean(plotAngles.T, axis=1)[:, None]), axis=1)
    R = 1 - residuals/tss
    print(R)


    fig3, axs3 = plt.subplots(numDoF)
    # plt.figure(0)
    # plt.title("Trained Model Performance")
    fig3.suptitle('Model Performance after Thresholding', fontsize=20, y=0.95, x=0.5)
    # fig.suptitle('Model Output in Blue; Reference Trajectory in Orange', fontsize=16)

    joints =  ['ThumbP', 'ThumbY', 'Index', 'Digits', 'Wrist flex']
    for i in range(numDoF):
        mus = FILE_NAMES[i]
        # plt.plot(t_goni, all_outs[i], label=f'Model Output {i}')
        # plt.plot(t_goni, np.array(plot_Angle)[:, i], label=f'Minimum Jerk {i}')
        axs3[i].plot(t_goni, newOuts[i], label=f'Model Output {i}', linewidth=2.5)
        axs3[i].plot(t_goni, np.array(plot_Angle)[:, i], label=f'Minimum Jerk {i}', linewidth=2)
        axs3[i].set_ylabel(joints[i], fontsize=12)
        axs3[i].autoscale(enable=True, axis='x', tight=True)

        # plt.plot(t_goni, preds[i], label=f'Predictions {i}')
        # plt.plot(t_goni, all_outs[i], label=f'Model Output {str(i)}')
        # plt.plot(t_goni, np.array(plot_Angle)[:, i], label=f'Minimum Jerk {str(i)}')

    axs3[numDoF - 1].set_xlabel('Time (s)', fontsize=12)
    axs3[0].set_title('Model Output in Blue; Reference Trajectory in Orange', fontsize=16)
    # plt.tight_layout()

    plt.show()
    #%% md
    # # Save the model
    #%%
    torch.save({'model_state_dict': system_dynamic_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                model_save_path)
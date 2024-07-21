import argparse
import math
import os
import yaml
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import numpy as np
from os.path import join
import wandb
import multiprocessing

from helpers.predict_utils import Config, get_data, train_model, get_dataNP


def wandb_process(arguments):
    config = arguments['config']
    wandb.agent(arguments['sweep_id'],
                lambda: train_model(arguments['trainsets'], arguments['testsets'], arguments['device'], config.wandb_mode, config.wandb_project, config.name))
    print(arguments['id'])


if __name__ == '__main__':

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using CUDA')
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print('Using MPS')
    else:
        device = torch.device("cpu")
        print('Using CPU')


    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.recordings]

    trainsets, testsets, combined_sets = get_dataNP(config, data_dirs, args.intact_hand, visualize=args.visualize)  # todo NP
    # invert the data:
    for set_id, train_set in enumerate(trainsets):
        train_set['data_myo'] = np.flip(train_set['data_myo'], axis=0).copy()
        train_set['data_angles'] = np.flip(train_set['data_angles'], axis=0).copy()
    for set_id, test_set in enumerate(testsets):
        test_set['data_myo'] = np.flip(test_set['data_myo'], axis=0).copy()
        test_set['data_angles'] = np.flip(test_set['data_angles'], axis=0).copy()
    for set_id, combined_set in enumerate(combined_sets):
        combined_set['data_myo'] = np.flip(combined_set['data_myo'], axis=0).copy()
        combined_set['data_angles'] = np.flip(combined_set['data_angles'], axis=0).copy()

    if args.hyperparameter_search:  # training on training set, evaluation on test set
        sweep_id = wandb.sweep(wandb_config, project=config.wandb_project)
        # wandb.agent(sweep_id, lambda: train_model(trainsets, testsets, device, config.wandb_mode, config.wandb_project, config.name))

        pool = multiprocessing.Pool(processes=4)
        pool.map(wandb_process, [{'id': i, 'config': config, 'sweep_id': sweep_id, 'trainsets': trainsets, 'testsets': testsets, 'device': device} for i in range(4)])



    if args.test:  # trains on the training set and saves the test set predictions
        model = train_model(trainsets, testsets, device, config.wandb_mode, config.wandb_project, config.name, config)

        for set_id, test_set in enumerate(testsets):
            test_set['data_myo'] = test_set['data_myo'][:len(test_set) // 4 * 4]  # todo NP
            test_set['data_angles'] = test_set['data_angles'][:len(test_set) // 4 * 4]  # todo NP

            val_pred = model.predict(test_set, config.features, config.targets).squeeze(0).to('cpu').detach().numpy()

            val_pred = np.clip(val_pred, -1, 1)
            test_set.loc[:, config.targets] = val_pred
            test_set = (test_set * math.pi + math.pi) / 2

            test_set.loc[:, (args.intact_hand, 'thumbInPlaneAng')] = test_set.loc[:, (args.intact_hand, 'thumbInPlaneAng')] - math.pi
            test_set.loc[:, (args.intact_hand, 'wristRot')] = (test_set.loc[:, (args.intact_hand, 'wristRot')] * 2) - math.pi
            test_set.loc[:, (args.intact_hand, 'wristFlex')] = (test_set.loc[:, (args.intact_hand, 'wristFlex')] - math.pi / 2)
            test_set.to_parquet(join(data_dirs[set_id], f'pred_angles-{args.config_name}.parquet'))


    if args.save_model:  # trains on the whole dataset and saves the model
        model = train_model(combined_sets, testsets, device, config.wandb_mode, config.wandb_project, config.name, config)

        model.to(torch.device('cpu'))
        args.person_dir = 'hvat' # todo NP
        os.makedirs(join('data', args.person_dir, 'models'), exist_ok=True)
        model.save(join('data', args.person_dir, 'models', f'{config.name}_inverted.pt')) # todo NP

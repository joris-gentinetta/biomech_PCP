import itertools

import pandas as pd

pd.options.mode.copy_on_write = True
idx = pd.IndexSlice
import multiprocessing

import warnings

warnings.filterwarnings("ignore")
from time import time
import argparse
import os
import yaml
import numpy as np
os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "1000000"

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import wandb
import torch
from torch.utils.data import DataLoader
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join
import math

torch.autograd.set_detect_anomaly(True)

from helpers.predict_utils import Config, OLDataset, evaluate_model, EarlyStopper, get_data, load_data


parser = argparse.ArgumentParser(description='Timeseries data analysis')
# parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32')
# parser.add_argument('-en', '--experiment_name', type=str, required=True, help='Experiment name')
parser.add_argument('--keep_states', action='store_true', help='Keep states')
args = parser.parse_args()


def online_train_model():
    with wandb.init():
        config = wandb.config

        epoch_len = 10

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Using CUDA')


        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     print('Using MPS')
        else:
            device = torch.device("cpu")
            print('Using CPU')

        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            print('TF32 enabled')

        if config.perturb:
            perturb_file = join('data', config.person_dir, 'online_trials', 'perturb', 'perturber.npy')
        else:
            perturb_file = join('data', config.person_dir, 'online_trials', 'non_perturb', 'perturber.npy')

        perturber = np.load(perturb_file)

        data_dirs = [join('data', config.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                     config.recordings]

        test_dirs = [join('data', config.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                     config.test_recordings] if config.test_recordings is not None else None

        _, valsets, combined_sets, testsets = get_data(config, data_dirs, args.intact_hand, visualize=False,
                                                      test_dirs=test_dirs, perturb_file=perturb_file)

        data_dir = join('data', config.person_dir, 'recordings', f'{config.online_file}', 'experiments', '1') # todo

        data = load_data(data_dir, args.intact_hand, config.features, perturber)
        trainsets = [data]

        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
                                           output_size=len(config.targets),
                                           **config)
        model.load(join('data', config.person_dir, 'models', f'{config.person_dir}.pt'))
        model.to(device)
        model.train()

        if config.model_type == 'ModularModel':
            epoch = 0
            # for param in model.model.activation_model.parameters():
            #     param.requires_grad = False if epoch < config.activation_model['n_freeze_epochs'] else True
            for param in model.model.muscle_model.parameters():
                param.requires_grad = False if epoch < config.muscle_model['n_freeze_epochs'] else True
            for param in model.model.joint_model.parameters():
                param.requires_grad = False if epoch < config.joint_model['n_freeze_epochs'] else True


        frame_id = 0

        angles_df = trainsets[0].loc[0, :].copy()
        train_dataset = OLDataset(trainsets, config.features, config.targets, device=device)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
        # train_dataloader = itertools.cycle(train_dataloader)
        train_dataloader = iter(train_dataloader)
        # early_stopper = EarlyStopper(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)

        best_val_loss = float('inf')

        angles_history = []
        emg_history = []
        states_history = []

        x, y, _ = next(train_dataloader)
        angles_history.append(y.squeeze(0).to('cpu').numpy())
        emg_history.append(x.squeeze(0).to('cpu').numpy())

        x, y, all_y = next(train_dataloader)
        angles_history.append(y.squeeze(0).to('cpu').numpy())
        emg_history.append(x.squeeze(0).to('cpu').numpy())

        angles_df.loc[:] = all_y.squeeze(0).to('cpu').numpy()


        if args.keep_states:
            # states = model.model.get_starting_states(1, torch.tensor(angles_history[-2:], dtype=torch.float32).unsqueeze(0).unsqueeze(1).to(device))
            history_samples = [1 for _ in range(config.batch_size)]
            x = np.concatenate([np.concatenate(
                [np.expand_dims(np.expand_dims(ah, axis=0), axis=1) for ah in angles_history[hs - 1: hs + 1]], axis=1)
                                for hs in history_samples], axis=0)
            x = torch.tensor(x, dtype=torch.float32).to(device)
            states = model.model.get_starting_states(config.batch_size, x)

            if config.model_type == 'ModularModel':
                states_history.append([states[2][st][-1:].detach() for st in range(3)])
                states_history.append([states[2][st][-1:].detach() for st in range(3)])
                states_history.append([states[2][st][-1:].detach() for st in range(3)])

            else:
                states_history.append(states[:, -1:, :].detach())
                states_history.append(states[:, -1:, :].detach())
                states_history.append(states[:, -1:, :].detach())


        print('Training model...')
        # with tqdm(range(1, config.n_epochs + 1)) as pbar:
        epoch = 0
        val_loss, test_loss, all_losses = evaluate_model(model, valsets, testsets, device, config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wandb.run.summary['best_epoch'] = epoch
            wandb.run.summary['best_val_loss'] = best_val_loss
        if test_loss < wandb.run.summary.get('best_test_loss', float('inf')):
            wandb.run.summary['best_test_loss'] = test_loss
            wandb.run.summary['best_test_epoch'] = epoch
        wandb.run.summary['used_epochs'] = epoch
        test_recording_names = config.test_recordings if config.test_recordings is not None else []
        log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
               enumerate(all_losses)}
        log['total_val_loss'] = val_loss
        log['total_test_loss'] = test_loss
        wandb.log(log, step=epoch)
        dataloader_empty = False

        # for epoch in pbar:
        for epoch in range(1, 100000):
            epoch_loss = 0

            # pbar.set_description(f'Epoch {epoch}')
            epoch_start_time = time()
            counter = 0
            for i in range(epoch_len):
                history_samples = np.random.randint(1, len(angles_history), size=config.batch_size-1)
                history_samples = np.array([history_sample for history_sample in history_samples] + [len(angles_history) - 1])
                if args.keep_states:
                    if config.model_type == 'ModularModel':
                        states = [None, None, [torch.concat([states_history[history_sample][0] for history_sample in history_samples], dim=0),
                                  torch.concat([states_history[history_sample][1] for history_sample in history_samples], dim=0),
                                    torch.concat([states_history[history_sample][2] for history_sample in history_samples], dim=0)]]

                    else:
                        states = torch.concat([states_history[history_sample] for history_sample in history_samples], dim=1)
                else:
                    x = np.concatenate([np.concatenate([np.expand_dims(np.expand_dims(ah, axis=0), axis=1)  for ah in angles_history[hs-1: hs+1]], axis=1) for hs in history_samples], axis=0)
                    x = torch.tensor(x, dtype=torch.float32).to(device)
                    states = model.model.get_starting_states(config.batch_size, x)

                seq_loss = 0
                #for x, y, all_y in train_dataloader:
                skip_update = False
                for s in range(config.seq_len):
                    try:
                        x, y, all_y = next(train_dataloader)
                    except:
                        dataloader_empty = True
                        break



                    start_time = time()
                    angles_history.append(y.squeeze(0).cpu().numpy())

                    # perturbed = perturber @ x.squeeze(0).cpu().numpy() # already done in load_data
                    perturbed = x.squeeze(0).cpu().numpy()

                    emg_history.append(perturbed)

                    y = torch.stack([torch.tensor(angles_history[history_sample], dtype=torch.float32) for history_sample in history_samples], dim=0).unsqueeze(1).to(device)
                    x = torch.stack([torch.tensor(emg_history[history_sample], dtype=torch.float32) for history_sample in history_samples], dim=0).unsqueeze(1).to(device)


                    outputs, states = model.model(x, states)
                    # print('y: ', y)
                    # print('outputs: ', outputs)
                    loss = model.criterion(outputs, y)
                    if torch.any(torch.isnan(loss)):
                        # print('NAN Loss!')
                        skip_update = True
                    else:
                        seq_loss = seq_loss + loss



                    history_samples = history_samples + 1

                    if args.keep_states:
                        if config.model_type == 'ModularModel':
                            history_states = [states[2][st].detach() for st in range(3)]
                            states_history.append([history_states[0][-1:], history_states[1][-1:], history_states[2][-1:]])
                            for hs in range(len(history_samples)-1):
                                states_history[history_samples[hs]] = [history_states[0][hs:hs+1], history_states[1][hs:hs+1], history_states[2][hs:hs+1]]

                        elif config.model_type != 'DenseNet':
                            history_states = states.detach()
                            states_history.append(history_states[:, -1:, :])

                            for hs in range(len(history_samples)-1):
                                states_history[history_samples[hs]] = history_states[:, hs:hs+1, :]

                if dataloader_empty:
                    break
                model.optimizer.zero_grad(set_to_none=True)
                if not skip_update:
                    seq_loss = seq_loss / config.seq_len
                    seq_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 4)
                    model.optimizer.step()
                else:
                    print('Skipping update')

                if epoch == 32:
                    for param_group in model.optimizer.param_groups: # todo
                        param_group['lr'] = 0.005
                if config.model_type == 'ModularModel':
                    states = [None, None, [states[2][st].detach() for st in range(3)]]
                elif config.model_type != 'DenseNet':
                    states = states.detach()

                wandb.log({'seq_loss': seq_loss.item()})
                epoch_loss = epoch_loss + seq_loss.item()
                trunctuator = 0
                seq_loss = 0

                if torch.any(torch.isnan(loss)):
                    print('NAN Loss!')

            val_loss, test_loss, all_losses = evaluate_model(model, valsets, testsets, device, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_val_loss'] = best_val_loss
            if test_loss < wandb.run.summary.get('best_test_loss', float('inf')):
                wandb.run.summary['best_test_loss'] = test_loss
                wandb.run.summary['best_test_epoch'] = epoch
            wandb.run.summary['used_epochs'] = epoch

            lr = model.scheduler.get_last_lr()[0]
            # model.scheduler.step(val_loss)  # Update the learning rate after each epoch
            epoch_loss = epoch_loss / epoch_len

            # pbar.set_postfix({'lr': lr, 'train_loss': epoch_loss, 'val_loss': val_loss})

            test_recording_names = config.test_recordings if config.test_recordings is not None else []
            log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
                   enumerate(all_losses)}
            log['total_val_loss'] = val_loss
            log['total_test_loss'] = test_loss
            log['train_loss'] = epoch_loss
            log['epoch'] = epoch
            log['lr'] = lr
            wandb.log(log)
            if dataloader_empty:
                break
        wandb.finish()




def wandb_process(sweep_id):
    wandb.agent(sweep_id,
                online_train_model)
    # print(arguments['id'])

if __name__ == '__main__':

    with open(join('data', 'offline_configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    sweep_id = wandb.sweep(wandb_config, project=config.wandb_project)
    # wandb.agent(sweep_id, online_train_model)

    pool = multiprocessing.Pool(processes=4)
    pool.map(wandb_process, [sweep_id for i in range(4)])


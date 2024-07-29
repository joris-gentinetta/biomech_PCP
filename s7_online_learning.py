import pandas as pd

pd.options.mode.copy_on_write = True
idx = pd.IndexSlice

import warnings

warnings.filterwarnings("ignore")
from time import time
import argparse
import os
import yaml

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import wandb
import torch
from torch.utils.data import DataLoader
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join

torch.autograd.set_detect_anomaly(True)
from multiprocessing import Queue as MPQueue

from helpers.predict_utils import Config, OLDataset, evaluate_model, EarlyStopper

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timeseries data analysis')
    parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
    parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
    parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32')
    parser.add_argument('-v', '--visualize', action='store_true', help='Plot data exploration results')
    parser.add_argument('-hs', '--hyperparameter_search', action='store_true', help='Perform hyperparameter search')
    parser.add_argument('-t', '--test', action='store_true', help='Test the model')
    parser.add_argument('-s', '--save_model', action='store_true', help='Save a model')
    parser.add_argument('--offline', action='store_true', help='Offline training')
    args = parser.parse_args()

    epoch_len = 1000
    sampling_frequency = 60
    calibration_frames = 30  # todo

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('Using CUDA')

        # List available GPUs
        if args.multi_gpu:
            n_gpus = torch.cuda.device_count()
            print(f'Number of available GPUs: {n_gpus}')
            for i in range(n_gpus):
                print(f'GPU{i}: {torch.cuda.get_device_name(i)}')

    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print('Using MPS')
    else:
        device = torch.device("cpu")
        print('Using CPU')

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print('TF32 enabled')

    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    emg_channels = [int(feature[1]) for feature in config.features]

    # data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
    #              config.recordings]
    #
    # test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
    #              config.test_recordings] if config.test_recordings is not None else None
    #
    # trainsets, testsets, combined_sets = get_data(config, data_dirs, args.intact_hand, visualize=args.visualize,
    #                                               test_dirs=test_dirs)
    #
    #
    #

    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    if args.offline:
        with wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config):
            config = wandb.config

            model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
                                               output_size=len(config.targets),
                                               **config)
            model.to(device)
            model.train()
            # if config.model_type == 'ActivationAndBiophys':  # todo
            #     for param in model.model.biophys_model.parameters():
            #         param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True

            train_dataset = OLDataset(trainsets, config.features, config.targets, device=device)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
            best_val_loss = float('inf')
            early_stopper = EarlyStopper(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)
            print('Training model...')
            with tqdm(range(config.n_epochs)) as pbar:
                for epoch in pbar:
                    pbar.set_description(f'Epoch {epoch}')
                    epoch_loss = 0
                    trunctuator = 0
                    states = model.model.get_starting_states(1, None)  # train_dataset[0][1].unsqueeze(0)
                    seq_loss = 0
                    for x, y in train_dataloader:

                        outputs, states = model.model(x, states)

                        loss = model.criterion(outputs, y)
                        seq_loss = seq_loss + loss

                        epoch_loss += loss.item()

                        trunctuator += 1
                        if trunctuator == config.seq_len:
                            model.optimizer.zero_grad(set_to_none=True)
                            seq_loss = seq_loss / config.seq_len
                            seq_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 4)
                            model.optimizer.step()
                            states = states.detach()
                            trunctuator = 0
                            seq_loss = 0

                        if torch.any(torch.isnan(loss)):
                            print('NAN Loss!')

                    val_loss, val_losses = evaluate_model(model, testsets, device, config)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wandb.run.summary['best_epoch'] = epoch
                        wandb.run.summary['best_val_loss'] = best_val_loss
                    wandb.run.summary['used_epochs'] = epoch

                    lr = model.scheduler.get_last_lr()[0]
                    model.scheduler.step(val_loss)  # Update the learning rate after each epoch
                    epoch_loss = epoch_loss / len(train_dataset)

                    pbar.set_postfix({'lr': lr, 'train_loss': epoch_loss, 'val_loss': val_loss})

                    # print('Total val loss:', val_loss)
                    log = {f'val_loss/{(config.recordings + config.test_recordings)[set_id]}': loss for set_id, loss in
                           enumerate(val_losses)}
                    log['total_val_loss'] = val_loss
                    log['train_loss'] = epoch_loss
                    log['lr'] = lr
                    wandb.log(log, step=epoch)

                    if early_stopper.early_stop(val_loss):
                        break

    else:

        wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config)
        config = wandb.config

        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
                                           output_size=len(config.targets),
                                           **config)
        model.to(device)
        model.train()
        # if config.model_type == 'ActivationAndBiophys':  # todo
        #     for param in model.model.biophys_model.parameters():
        #         param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True

        from online_utils import JointsProcess, AnglesProcess, VisualizeProcess

        queue_size = 5
        jointsProcess = JointsProcess(args.intact_hand, queue_size)
        anglesProcess = AnglesProcess(args.intact_hand, queue_size, jointsProcess.outputQ)
        visualizeQueue = MPQueue(queue_size)
        visualizeProcess = VisualizeProcess(args.intact_hand, visualizeQueue)

        jointsProcess.start()
        anglesProcess.start()
        visualizeProcess.start()

        jointsProcess.initialized.wait()
        anglesProcess.initialized.wait()
        visualizeProcess.initialized.wait()

        frame_id = 0

        print('Training model...')
        best_val_loss = float('inf')
        with tqdm(range(config.n_epochs)) as pbar:
            for epoch in pbar:
                pbar.set_description(f'Epoch {epoch}')
                epoch_loss = 0
                trunctuator = 0
                states = model.model.get_starting_states(1, None)  # train_dataset[0][1].unsqueeze(0)
                seq_loss = 0
                check_time = time()
                for i in range(epoch_len):
                    print(1 / (time() - check_time))
                    check_time = time()

                    angles_df, emg_timestep = anglesProcess.outputQ.get()

                    y = torch.tensor(angles_df.loc[frame_id, config.targets], dtype=torch.float32).unsqueeze(
                        0).unsqueeze(0).to(device)

                    x = torch.tensor(emg_timestep[emg_channels], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(
                        device)

                    outputs, states = model.model(x, states)
                    # print('y: ', y)
                    # print('outputs: ', outputs)
                    loss = model.criterion(outputs, y)
                    seq_loss = seq_loss + loss

                    epoch_loss += loss.item()
                    pred_angels_df = angles_df.copy()
                    pred_angels_df.loc[frame_id, config.targets] = outputs.squeeze().to('cpu').detach().numpy()
                    if not visualizeQueue.full():
                        visualizeQueue.put((angles_df, pred_angels_df))
                    else:
                        print('VisualizeProcess input queue full')

                    trunctuator += 1
                    if trunctuator == config.seq_len:
                        model.optimizer.zero_grad(set_to_none=True)
                        seq_loss = seq_loss / config.seq_len
                        seq_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.model.parameters(), 4)
                        model.optimizer.step()
                        states = states.detach()
                        wandb.log({'seq_loss': seq_loss.item()})
                        trunctuator = 0
                        seq_loss = 0

                    if torch.any(torch.isnan(loss)):
                        print('NAN Loss!')

                # val_loss, val_losses = evaluate_model(model, testsets, device, config)
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     wandb.run.summary['best_epoch'] = epoch
                #     wandb.run.summary['best_val_loss'] = best_val_loss
                # wandb.run.summary['used_epochs'] = epoch
                #
                # lr = model.scheduler.get_last_lr()[0]
                # model.scheduler.step(val_loss)  # Update the learning rate after each epoch
                # epoch_loss = epoch_loss / epoch_len
                # pbar.set_postfix({'lr': lr, 'train_loss': epoch_loss, 'val_loss': val_loss})
                #
                # # print('Total val loss:', val_loss)
                # log = {f'val_loss/{(config.recordings + config.test_recordings)[set_id]}': loss for set_id, loss in
                #        enumerate(val_losses)}
                # log['total_val_loss'] = val_loss
                # log['train_loss'] = epoch_loss
                # log['lr'] = lr
                # wandb.log(log, step=epoch)

import pandas as pd

pd.options.mode.copy_on_write = True
idx = pd.IndexSlice

import warnings

warnings.filterwarnings("ignore")
from time import time
import argparse
import os
import yaml
import numpy as np

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import wandb
import torch
from torch.utils.data import DataLoader
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join
import signal
import cv2
torch.autograd.set_detect_anomaly(True)
from multiprocessing import Queue as MPQueue

from helpers.predict_utils import Config, OLDataset, evaluate_model, EarlyStopper, get_data
from online_utils import JointsProcess, AnglesProcess, VisualizeProcess, ProcessManager, InputThread, SaveThread


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Timeseries data analysis')
    parser.add_argument('--person_dir', type=str, required=True, help='Person directory')
    parser.add_argument('--intact_hand', type=str, required=True, help='Intact hand (Right/Left)')
    parser.add_argument('--config_name', type=str, required=True, help='Training configuration')
    parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32')
    parser.add_argument('-s', '--save_model', action='store_true', help='Save a model') # todo
    parser.add_argument('--offline', action='store_true', help='Offline training')
    parser.add_argument('-v', '--visualize', action='store_true', help='Visualize hand movements')
    parser.add_argument('-en', '--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('-c', '--camera', type=int, required=False, help='Camera id')
    parser.add_argument('-si', '--save_input', action='store_true', help='Save input data')
    parser.add_argument('-cf', '--calibration_frames', type=int, default=20, help='Number of calibration frames')
    args = parser.parse_args()

    save_path = join('data', args.person_dir, 'recordings', args.experiment_name)
    if os.path.exists(save_path) and args.save_input: # todo
        print('Experiment already exists!')
        exit()
    os.makedirs(save_path, exist_ok=True)

    processManager = ProcessManager()
    signal.signal(signal.SIGINT, processManager.signal_handler)


    epoch_len = 10
    # sampling_frequency = 60
    calibration_frames = 30  # todo
    queue_size = 50

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

    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    emg_channels = [int(feature[1]) for feature in config.features]



    with open(join('data', args.person_dir, 'configs', f'{args.config_name}.yaml'), 'r') as file:
        wandb_config = yaml.safe_load(file)
        config = Config(wandb_config)

    data_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.recordings]

    test_dirs = [join('data', args.person_dir, 'recordings', recording, 'experiments', '1') for recording in
                 config.test_recordings] if config.test_recordings is not None else None

    trainsets, testsets, combined_sets = get_data(config, data_dirs, args.intact_hand, visualize=False,
                                                      test_dirs=test_dirs)

    if args.save_input:
        temp_vc = cv2.VideoCapture(args.camera)
        width = temp_vc.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = temp_vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
        temp_vc.release()

        vc = InputThread(src=args.camera, queueSize=queue_size, save=True)
        processManager.manage_process(vc)
        st = SaveThread(vc.outputQ, frame_size=(int(width), int(height)), save_path=save_path)
        processManager.manage_process(st)


    elif args.offline:


        if args.visualize:
            visualizeQueue = MPQueue(queue_size)
            visualizeProcess = VisualizeProcess(args.intact_hand, visualizeQueue)
            processManager.manage_process(visualizeProcess)


        with wandb.init(mode=config.wandb_mode, project=config.wandb_project, name=config.name, config=config):
            config = wandb.config

            model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features),
                                               output_size=len(config.targets),
                                               **config)
            model.load(join('data', args.person_dir, 'models', f'{config.name}.pt'))
            model.to(device)
            model.train()
            # if config.model_type == 'ActivationAndBiophys':  # todo
            #     for param in model.model.biophys_model.parameters():
            #         param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True
            angles_df = trainsets[0].loc[0, :].copy()
            train_dataset = OLDataset(trainsets, config.features, config.targets, device=device)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
            best_val_loss = float('inf')
            early_stopper = EarlyStopper(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)
            print('Training model...')
            with tqdm(range(1, config.n_epochs + 1)) as pbar:
                epoch = 0
                val_loss, val_losses = evaluate_model(model, testsets, device, config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_val_loss'] = best_val_loss
                wandb.run.summary['used_epochs'] = epoch
                test_recording_names = config.test_recordings if config.test_recordings is not None else []
                log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
                       enumerate(val_losses)}
                log['total_val_loss'] = val_loss
                wandb.log(log, step=epoch)

                for epoch in pbar:
                    pbar.set_description(f'Epoch {epoch}')
                    epoch_loss = 0
                    trunctuator = 0
                    states = model.model.get_starting_states(1, None)  # train_dataset[0][1].unsqueeze(0)
                    seq_loss = 0
                    time_1 = time()
                    model.train()

                    for x, y, all_y in train_dataloader:
                        # print(1 / (time() - time_1))
                        time_1 = time()


                        outputs, states = model.model(x, states)
                        angles_df.loc[:] = all_y.squeeze(0).numpy()
                        pred_angels_df = angles_df.copy()
                        pred_angels_df.loc[config.targets] = outputs.squeeze().to('cpu').detach().numpy()

                        if args.visualize and not visualizeQueue.full():
                            visualizeQueue.put((angles_df, pred_angels_df))
                        elif args.visualize:
                            print('VisualizeProcess input queue full')
                            print('VisualizeProcess fps: ', visualizeProcess.fps.value)
                            pass


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

                    test_recording_names = config.test_recordings if config.test_recordings is not None else []
                    log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
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
        model.load(join('data', args.person_dir, 'models', f'{config.name}.pt'))
        model.to(device)
        model.train()
        # if config.model_type == 'ActivationAndBiophys':  # todo
        #     for param in model.model.biophys_model.parameters():
        #         param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True

        if args.visualize:
            visualizeQueue = MPQueue(queue_size)
            visualizeProcess = VisualizeProcess(args.intact_hand, visualizeQueue)
            processManager.manage_process(visualizeProcess)

        joint_to_angles_Q = MPQueue(queue_size)
        anglesProcess = AnglesProcess(args.intact_hand, queue_size, joint_to_angles_Q)
        jointsProcess = JointsProcess(args.intact_hand, queue_size, joint_to_angles_Q, save_path, args.camera, args.save_input, args.calibration_frames)
        processManager.manage_process(jointsProcess)
        processManager.manage_process(anglesProcess)



        frame_id = 0

        print('Training model...')
        best_val_loss = float('inf')
        fps = 0
        true_start_time = time()


        for i in range(60):
            _, _ = anglesProcess.outputQ.get(timeout=2)

        angles_history = []
        emg_history = []
        states_history = []

        states = model.model.get_starting_states(1, None)
        angles_df, emg_timestep = anglesProcess.outputQ.get(timeout=2)

        angles_history.append(angles_df.loc[config.targets].values)
        emg_history.append(emg_timestep[emg_channels])
        states_history.append(states.detach())

        states_history.append(states.detach())

        with tqdm(range(config.n_epochs)) as pbar:
            for epoch in pbar:
                epoch_loss = 0

                pbar.set_description(f'Epoch {epoch}')
                epoch_start_time = time()
                counter = 0
                for i in range(epoch_len):
                    history_samples = np.random.randint(0, len(angles_history), size=config.batch_size-1)
                    history_samples = np.array([history_sample for history_sample in history_samples] + [len(angles_history) - 1])
                    states = torch.concat([states_history[history_sample] for history_sample in history_samples], dim=1)

                    # states = states
                    for s in range(config.seq_len):
                        seq_loss = 0

                        angles_df, emg_timestep = anglesProcess.outputQ.get(timeout=2)
                        start_time = time()
                        angles_history.append(angles_df.loc[config.targets].values)
                        emg_history.append(emg_timestep[emg_channels])

                        y = torch.stack([torch.tensor(angles_history[history_sample], dtype=torch.float32) for history_sample in history_samples], dim=0).unsqueeze(1).to(device)
                        x = torch.stack([torch.tensor(emg_history[history_sample], dtype=torch.float32) for history_sample in history_samples], dim=0).unsqueeze(1).to(device)


                        outputs, states = model.model(x, states)
                        # print('y: ', y)
                        # print('outputs: ', outputs)
                        loss = model.criterion(outputs, y)
                        seq_loss = seq_loss + loss

                        epoch_loss += loss.item()
                        pred_angels_df = angles_df.copy()
                        pred_angels_df.loc[config.targets] = outputs[-1].squeeze().to('cpu').detach().numpy()
                        if args.visualize and not visualizeQueue.full():
                            visualizeQueue.put((angles_df, pred_angels_df))
                            counter += 1
                        elif args.visualize:
                            print('VisualizeProcess input queue full')
                            print('VisualizeProcess fps: ', visualizeProcess.fps.value)
                            pass

                        history_samples = history_samples + 1

                        history_states = states.detach()
                        states_history.append(history_states[:, -1, :].unsqueeze(1))

                        for hs in range(len(history_samples)-1):
                            states_history[history_samples[hs]] = history_states[:, hs, :].unsqueeze(1)




                    model.optimizer.zero_grad(set_to_none=True)
                    seq_loss = seq_loss / config.seq_len
                    seq_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 4)
                    model.optimizer.step()
                    # states = states.detach()
                    wandb.log({'seq_loss': seq_loss.item()})
                    trunctuator = 0
                    seq_loss = 0

                    if torch.any(torch.isnan(loss)):
                        print('NAN Loss!')
                    fps = 1 / (time() - start_time)
                    true_end_time = time()
                    true_fps = 1 / (true_end_time - true_start_time)
                    true_start_time = true_end_time

                    print_fps = False
                    if print_fps:
                        print(f'True fps: {true_fps}')
                        print('InputThread fps: ', jointsProcess.input_fps.value)
                        print('JointsProcess fps: ', jointsProcess.fps.value)
                        print('AnglesProcess fps: ', anglesProcess.fps.value)
                        if args.visualize:
                            print('VisualizeProcess fps: ', visualizeProcess.fps.value)
                        print('FPS: ', fps)
                        if trunctuator == 0:
                            print('#################### FPS: ', fps)
                epoch_end_time = time()
                print('average epoch fps: ', counter / (epoch_end_time - epoch_start_time))

                # val_loss, val_losses = evaluate_model(model, testsets, device, config)
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     wandb.run.summary['best_epoch'] = epoch
                #     wandb.run.summary['best_val_loss'] = best_val_loss
                # wandb.run.summary['used_epochs'] = epoch
                #
                # lr = model.scheduler.get_last_lr()[0]
                # # model.scheduler.step(val_loss)  # Update the learning rate after each epoch
                # epoch_loss = epoch_loss / epoch_len
                # pbar.set_postfix({'lr': lr, 'train_loss': epoch_loss, 'val_loss': val_loss})
                #
                # # print('Total val loss:', val_loss)
                # test_recording_names = config.test_recordings if config.test_recordings is not None else []
                # log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in
                #        enumerate(val_losses)}
                # log['total_val_loss'] = val_loss
                # log['train_loss'] = epoch_loss
                # log['lr'] = lr
                # log['epoch'] = epoch
                # wandb.log(log)
                # model.train()
                model.save(join('data', args.person_dir, 'models', f'{config.name}_online.pt'))

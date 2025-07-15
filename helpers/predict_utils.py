import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import random
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import math

from helpers.utils import restore_multiindex_columns


def scale_data(data, intact_hand):
    data.loc[:, (intact_hand, 'thumbInPlaneAng')] = data.loc[:,(intact_hand, 'thumbInPlaneAng')] + math.pi
    data.loc[:, (intact_hand, 'wristRot')] = (data.loc[:, (intact_hand, 'wristRot')] + math.pi) / 2
    data.loc[:, (intact_hand, 'wristFlex')] = (data.loc[:, (intact_hand, 'wristFlex')] + math.pi / 2)
    data = (2 * data - math.pi) / math.pi
    data = data.clip(-1, 1)
    return data

def rescale_data(angles_df, intact_hand):
    series = False
    if isinstance(angles_df, pd.Series):
        series = True
        angles_df = angles_df.to_frame().T
    angles_df = angles_df.clip(-1, 1)
    angles_df = (angles_df * math.pi + math.pi) / 2
    angles_df.loc[:, (intact_hand, 'wristFlex')] = angles_df.loc[:, (intact_hand, 'wristFlex')] - math.pi / 2
    angles_df.loc[:, (intact_hand, 'wristRot')] = (angles_df.loc[:, (intact_hand, 'wristRot')] * 2) - math.pi
    angles_df.loc[:, (intact_hand, 'thumbInPlaneAng')] = angles_df.loc[:, (intact_hand, 'thumbInPlaneAng')] - math.pi
    if series:
        return angles_df.iloc[0]
    else:
        return angles_df

def load_data(data_dir, intact_hand, features, perturber=None):
    angles = pd.read_parquet(join(data_dir, 'cropped_smooth_angles.parquet'))
    angles = restore_multiindex_columns(angles)
    angles.index = range(len(angles))

    try:
        emg = np.load(join(data_dir, 'cropped_aligned_emg.npy'))
    except:
        emg = np.load(join(data_dir, 'cropped_emg.npy'))

    data = angles.copy()
    data = scale_data(data, intact_hand)

    int_features = [int(feature[1]) for feature in features]
    emg = emg[:, int_features]
    if perturber is not None:
        emg = (perturber @ emg.T).T

    for feature in features:
        data[tuple(feature)] = emg[:, int_features.index(int(feature[1]))]

    return data


def get_data(config, data_dirs, intact_hand, visualize=False, test_dirs=None, perturb_file=None):

    trainsets = []
    valsets = []
    testsets = []
    combined_sets = []
    perturber = np.load(perturb_file) if perturb_file is not None else None
    for recording_id, data_dir in enumerate(data_dirs):
        data = load_data(data_dir, intact_hand, config.features, perturber)

        # temp
        data = data.loc[60:].copy()

        if visualize:
            # axs = data[config.features].plot(subplots=True, ylim=(-0.1, 1.1))
            # for ax in axs: ax.legend(loc='upper right')
            # plt.suptitle(f'Features {config.recordings[recording_id]}')
            # plt.show()

            # axs = data[config.targets].plot(subplots=True, ylim=(-1.1, 1.1))
            # for ax in axs: ax.legend(loc='upper right')
            # plt.suptitle(f'Targets {config.recordings[recording_id]}')
            # plt.show()

            leftSubplots = len(config.features)
            rightSubplots = len(config.targets)
            fig, axs = plt.subplots(max(leftSubplots, rightSubplots), 2, figsize=(15, 16))
            fig.suptitle(f'{config.recordings[recording_id]}')
            for i, feature in enumerate(config.features):
                axs[i, 0].plot(data[feature])
                # axs[i, 0].set_title(feature)
                axs[i, 0].set_ylim(-0.1, 1.1)
                axs[i, 0].set_ylabel(feature[1])

            for i, target in enumerate(config.targets):
                axs[i, 1].plot(data[target])
                # axs[i, 1].set_title(target)
                axs[i, 1].set_ylim(-1.1, 1.1)
                axs[i, 1].yaxis.set_label_position('right')
                axs[i, 1].set_ylabel(target[1])
            plt.tight_layout()
            plt.show()


        # if test_dirs is None:
        test_set = data.loc[len(data) // 5 * 4:].copy()
        train_set = data.loc[:len(data) // 5 * 4].copy()
        trainsets.append(train_set)
        valsets.append(test_set)
        combined_sets.append(data.copy())
        # else:
        #     train_set = data.copy()
        #     trainsets.append(train_set)
        #     combined_sets.append(train_set)

    if test_dirs is not None:
        for test_dir in test_dirs:
            data = load_data(test_dir, intact_hand, config.features, perturber)
            test_set = data.loc[len(data) // 5 * 4:].copy()
            train_set = data.loc[: len(data) // 5 * 4].copy()
            testsets.append(test_set)
            # combined_sets.append(data)


    return trainsets, valsets, combined_sets, testsets


def train_model(trainsets, valsets, testsets, device, wandb_mode, wandb_project, wandb_name, config=None, person_dir='test'):
    with wandb.init(mode=wandb_mode, project=wandb_project, name=wandb_name, config=config):
        config = wandb.config

        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets), **config)
        model.to(device)

        dataset = TSDataset(trainsets, config.features, config.targets, seq_len=config.seq_len, device=device)
        dataloader = TSDataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

        best_val_loss = float('inf')
        early_stopper = EarlyStopper(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)
        print('Training model...')
        with tqdm(range(model.n_epochs)) as pbar:
            for epoch in pbar:
                pbar.set_description(f'Epoch {epoch}')

                if config.model_type == 'ModularModel':  # todo
                    for param in model.model.activation_model.parameters():
                        param.requires_grad = False if epoch < config.activation_model['n_freeze_epochs'] else True
                    for param in model.model.muscle_model.parameters():
                        param.requires_grad = False if epoch < config.muscle_model['n_freeze_epochs'] else True
                    for param in model.model.joint_model.parameters():
                        param.requires_grad = False if epoch < config.joint_model['n_freeze_epochs'] else True

                train_loss = model.train_one_epoch(dataloader)

                val_loss, test_loss, val_losses = evaluate_model(model, valsets, testsets, device, config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_val_loss'] = best_val_loss
                if test_loss < wandb.run.summary.get('best_test_loss', float('inf')):
                    wandb.run.summary['best_test_loss'] = test_loss
                    wandb.run.summary['best_test_epoch'] = epoch
                    model.save('/tmp/bestWeights.pt')
                wandb.run.summary['used_epochs'] = epoch

                lr = model.scheduler.get_last_lr()[0]
                if epoch > 15: # todo
                    model.scheduler.step(val_loss)  # Update the learning rate after each epoch #todo train or val loss
                pbar.set_postfix({'lr': lr, 'train_loss': train_loss, 'val_loss': val_loss, 'test_loss': test_loss})

                # print('Total val loss:', val_loss)
                test_recording_names = config.test_recordings if config.test_recordings is not None else []
                log = {f'val_loss/{(config.recordings + test_recording_names)[set_id]}': loss for set_id, loss in enumerate(val_losses)}
                log['total_val_loss'] = val_loss
                log['total_test_loss'] = test_loss
                log['train_loss'] = train_loss
                log['lr'] = lr
                wandb.log(log, step=epoch)

                if early_stopper.early_stop(val_loss):
                    break
        # model.save(join('data', person_dir, 'models', f'{wandb_name}_bs{config.batch_size}_sl{config.seq_len}.pt'))
        model.load('/tmp/bestWeights.pt')
        return model


def evaluate_model(model, valsets, testsets, device, config):
    warmup_steps = config.warmup_steps # todo
    # warmup_steps = config.seq_len - 1
    val_losses = []
    for set_id, val_set in enumerate(valsets):
        val_pred = model.predict(val_set, config.features, config.targets).squeeze(0)
        loss = model.criterion(val_pred[warmup_steps:],
                                    torch.tensor(val_set[config.targets].values, dtype=torch.float32)[
                                    warmup_steps:].to(device))
        loss = float(loss.to('cpu').detach())
        val_losses.append(loss)
    total_val_loss = sum(val_losses) / len(val_losses)

    test_losses = []
    for set_id, test_set in enumerate(testsets):
        test_pred = model.predict(test_set, config.features, config.targets).squeeze(0)
        loss = model.criterion(test_pred[warmup_steps:],
                                    torch.tensor(test_set[config.targets].values, dtype=torch.float32)[
                                    warmup_steps:].to(device))
        loss = float(loss.to('cpu').detach())
        test_losses.append(loss)
    total_test_loss = sum(test_losses) / len(test_losses)

    return total_val_loss, total_test_loss, val_losses + test_losses


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        # elif validation_loss > (self.min_validation_loss + self.min_delta):
        elif validation_loss > (self.min_validation_loss*(1 + self.min_delta)):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Config:
    def __init__(self, dictionary):
        setattr(self, 'name', dictionary['name'])
        for key, value in dictionary['parameters'].items():
            for key_2, value_2 in value.items():
                if isinstance(value_2, list):
                    for id, value_3 in enumerate(value_2):
                        if isinstance(value_3, list):
                            value_2[id] = tuple(value_3)
                setattr(self, key, value_2)

    def to_dict(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith('_')}


class TSDataset(Dataset):
    def __init__(self, data_sources, features, targets, seq_len, device, index_shift=0, dummy_labels=False):
        self.data_sources = data_sources
        for i in range(len(self.data_sources)):
            self.data_sources[i] = self.data_sources[i].astype('float32')
            self.data_sources[i] = self.data_sources[i].reset_index(drop=True)
        self.features = features
        self.targets = targets
        self.seq_len = seq_len
        self.index_shift = index_shift
        # self.lengths = [len(data) - 2 * self.seq_len for data in self.data_sources]
        self.lengths = [len(data) // self.seq_len - 1 for data in self.data_sources]
        self.starts = [0] + [sum(self.lengths[:i]) for i in range(1, len(self.lengths))]
        self.dummy_labels = dummy_labels
        self.device = device

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        set_idx = -1
        for start in self.starts:
            if idx < start:
                break
            set_idx += 1
        idx = idx - self.starts[set_idx]
        x = torch.tensor(self.data_sources[set_idx].loc[idx * self.seq_len + self.index_shift: (idx + 1) * self.seq_len + self.index_shift - 1, self.features].values, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data_sources[set_idx].loc[idx * self.seq_len + self.index_shift: (idx + 1) * self.seq_len + self.index_shift - 1, self.targets].values, dtype=torch.float32, device=self.device)
        if self.dummy_labels:
            l = torch.ones_like(y, dtype=torch.float32, device=self.device)
            return x, y, l
        else:
            return x, y


    def set_index_shift(self, shift):
        self.index_shift = shift


class OLDataset(Dataset):
    def __init__(self, data_sources, features, targets, device):
        self.data_sources = data_sources
        for i in range(len(self.data_sources)):
            self.data_sources[i] = self.data_sources[i].astype('float32')
            self.data_sources[i] = self.data_sources[i].reset_index(drop=True)
        # combine all data sources:
        self.data = pd.concat(self.data_sources, axis=0)
        self.data = self.data.reset_index(drop=True)
        self.features = features
        self.targets = targets
        self.device = device

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        x = torch.tensor(self.data.loc[idx, self.features].values, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data.loc[idx, self.targets].values, dtype=torch.float32, device=self.device)

        return x, y, self.data.loc[idx, :].values


class TSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_index_shift(random.randint(0, self.dataset.seq_len))
        return super().__iter__()

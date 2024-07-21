import os

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import random
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def get_data(config, data_dirs, intact_hand, visualize=False):

    trainsets = []
    testsets = []
    combined_sets = []
    for recording_id, data_dir in enumerate(data_dirs):
        angles = pd.read_parquet(join(data_dir, 'cropped_smooth_angles.parquet'))
        angles.index = range(len(angles))
        emg = np.load(join(data_dir, 'cropped_aligned_emg.npy'))

        data = angles.copy()
        data.loc[:, (intact_hand, 'thumbInPlaneAng')] = data.loc[:,
                                                             (intact_hand, 'thumbInPlaneAng')] + math.pi
        data.loc[:, (intact_hand, 'wristRot')] = (data.loc[:, (intact_hand, 'wristRot')] + math.pi) / 2
        data.loc[:, (intact_hand, 'wristFlex')] = (data.loc[:, (intact_hand, 'wristFlex')] + math.pi / 2)

        data = (2 * data - math.pi) / math.pi
        data = np.clip(data, -1, 1)

        for feature in config.features:
            data[feature] = emg[:, int(feature[1])]

        if visualize:
            data[config.features].plot(subplots=True)
            plt.title(f'Features {config.recordings[recording_id]}')
            plt.show()

            data[config.targets].plot(subplots=True)
            plt.title('Targets')
            plt.show()

        test_set = data.loc[len(data) // 5 * 4:].copy()
        train = data.loc[: len(data) // 5 * 4].copy()
        trainsets.append(train)
        testsets.append(test_set)
        combined_sets.append(data.copy())

    return trainsets, testsets, combined_sets


def get_dataNP(config, data_dirs, intact_hand, visualize=False):
    data_folder = '/Users/jg/projects/biomech/BCI_ALVI_challenge/dataset_v2_blocks/amputant/left/fedya_tropin_standart_elbow_left/preproc_angles'
    trainsets = []
    combined_sets = []
    for d in os.listdir(join(data_folder, 'train')):
        trainsets.append(dict(np.load(join(data_folder, 'train', d))))
        combined_sets.append(dict(np.load(join(data_folder, 'train', d))))

    testsets = []
    for d in os.listdir(join(data_folder, 'test')):
        testsets.append(dict(np.load(join(data_folder, 'test', d))))
        combined_sets.append(dict(np.load(join(data_folder, 'test', d))))

    # train_dataset = TSDatasetNP(train_data, seq_len=125, device='cpu')
    # train_dataloader = TSDataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    return trainsets, testsets, combined_sets


def train_model(trainsets, testsets, device, wandb_mode, wandb_project, wandb_name, config=None):
    with wandb.init(mode=wandb_mode, project=wandb_project, name=wandb_name, config=config):
        config = wandb.config

        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets), **config)
        model.to(device)

        dataset = TSDatasetNP(trainsets, config.features, config.targets, seq_len=config.seq_len, device=device) # todo NP
        dataloader = TSDataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

        best_val_loss = float('inf')
        early_stopper = EarlyStopper(patience=config.early_stopping_patience, min_delta=config.early_stopping_delta)
        print('Training model...')
        with tqdm(range(model.n_epochs)) as pbar:
            for epoch in pbar:
                pbar.set_description(f'Epoch {epoch}')

                if config.model_type == 'ActivationAndBiophys': # todo
                    for param in model.model.biophys_model.parameters():
                        param.requires_grad = False if epoch < config.biophys_config['n_freeze_epochs'] else True

                train_loss = model.train_one_epoch(dataloader)

                val_loss, val_losses = evaluate_modelNP(model, testsets, device, config)  # todo NP
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_val_loss'] = best_val_loss
                wandb.run.summary['used_epochs'] = epoch


                lr = model.scheduler.get_last_lr()[0]
                model.scheduler.step(val_loss)  # Update the learning rate after each epoch
                pbar.set_postfix({'lr': lr, 'train_loss': train_loss, 'val_loss': val_loss})

                # print('Total val loss:', val_loss)
                log = {}  # todo NP
                # log = {f'val_loss/{config.recordings[set_id]}': loss for set_id, loss in enumerate(val_losses)} # todo NP
                log['total_val_loss'] = val_loss
                log['train_loss'] = train_loss
                log['lr'] = lr
                wandb.log(log, step=epoch)

                if early_stopper.early_stop(val_loss):
                    break

        return model


def evaluate_model(model, testsets, device, config):
    losses = []
    for set_id, test_set in enumerate(testsets):
        test_set = test_set[:len(test_set) // 4 * 4] # todo
        val_pred = model.predict(test_set, config.features, config.targets).squeeze(0)
        loss = model.criterion(val_pred[config.warmup_steps:],
                                    torch.tensor(test_set[config.targets].values, dtype=torch.float32)[
                                    config.warmup_steps:].to(device))
        loss = float(loss.to('cpu').detach())
        losses.append(loss)
    total_loss = sum(losses) / len(losses)

    return total_loss, losses


def evaluate_modelNP(model, testsets, device, config):
    losses = []
    for set_id, test_set in enumerate(testsets):
        test_set['data_myo'] = test_set['data_myo'][:len(test_set) // 4 * 4]
        test_set['data_angles'] = test_set['data_angles'][:len(test_set) // 4 * 4]
        val_pred = model.predictNP(test_set, config.features, config.targets).squeeze(0)
        loss = model.criterion(val_pred[config.warmup_steps:], torch.tensor(test_set['data_angles'][config.warmup_steps:], dtype=torch.float32).to(device))
        loss = float(loss.to('cpu').detach())
        losses.append(loss)
    total_loss = sum(losses) / len(losses)

    return total_loss, losses





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
        elif validation_loss > (self.min_validation_loss + self.min_delta):
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


class TSDatasetNP(Dataset):
    def __init__(self, data_sources, features, targets, seq_len, device, index_shift=0, dummy_labels=False):
        self.data_sources = data_sources
        self.seq_len = seq_len
        self.index_shift = index_shift
        # self.lengths = [len(data) - 2 * self.seq_len for data in self.data_sources]
        self.lengths = [len(data['data_myo']) // self.seq_len - 1 for data in self.data_sources]
        self.starts = [0] + [sum(self.lengths[:i]) for i in range(1, len(self.lengths))]
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
        x = torch.tensor(self.data_sources[set_idx]['data_myo'][idx * self.seq_len + self.index_shift: (idx + 1) * self.seq_len + self.index_shift - 1, :], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data_sources[set_idx]['data_angles'][idx * self.seq_len + self.index_shift: (idx + 1) * self.seq_len + self.index_shift - 1, :], dtype=torch.float32, device=self.device)

        return x, y


    def set_index_shift(self, shift):
        self.index_shift = shift


class TSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_index_shift(random.randint(0, self.dataset.seq_len))
        return super().__iter__()


if __name__ == '__main__':
    data_folder = '/Users/jg/projects/biomech/BCI_ALVI_challenge/dataset_v2_blocks/amputant/left/fedya_tropin_standart_elbow_left/preproc_angles'
    train_data = []
    for d in os.listdir(join(data_folder, 'train')):
        train_data.append(dict(np.load(join(data_folder, 'train', d))))


    train_dataset = TSDatasetNP(train_data, seq_len=125, device='cpu')
    train_dataloader = TSDataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    for x, y in train_dataloader:
        print(x.shape, y.shape)
        break
    print()



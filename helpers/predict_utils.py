import wandb
import torch
import os
from torch.utils.data import Dataset, DataLoader
import random
from helpers.models import TimeSeriesRegressorWrapper
from tqdm import tqdm
from os.path import join
import numpy as np
import pandas as pd
import config
import matplotlib.pyplot as plt
import math

def scale_data(data, intact_hand):
    # # If in-plane thumb angle exists, shift by pi
    # col = (intact_hand, 'thumbInPlaneAng')
    # if col in data.columns:
    #     data.loc[:, col] = data.loc[:, col] + math.pi
    # col = (intact_hand, 'wristRot')
    # if col in data.columns:
    #     data.loc[:, col] = (data.loc[:, col] + math.pi) / 2
    # col = (intact_hand, 'wristFlex')
    # if col in data.columns:
    #     data.loc[:, col] = (data.loc[:, col] + math.pi) / 2
    # # data.loc[:, (intact_hand, 'thumbInPlaneAng')] = data.loc[:,(intact_hand, 'thumbInPlaneAng')] + math.pi
    # # data.loc[:, (intact_hand, 'wristRot')] = (data.loc[:, (intact_hand, 'wristRot')] + math.pi) / 2
    # # data.loc[:, (intact_hand, 'wristFlex')] = (data.loc[:, (intact_hand, 'wristFlex')] + math.pi / 2)
    # data = (2 * data - math.pi) / math.pi
    # data = data.clip(-1, 1)
    # return data
    scaled = data.copy()
    # Angles in [0, 120]
    angle_cols = [
        (intact_hand, 'index_Pos'),
        (intact_hand, 'middle_Pos'),
        (intact_hand, 'ring_Pos'),
        (intact_hand, 'pinky_Pos'),
        (intact_hand, 'thumbFlex_Pos'),
    ]
    for col in angle_cols:
        if col in scaled.columns:
            scaled[col] = (scaled[col] - 60) / 60
            scaled[col] = scaled[col].clip(-1, 1)
    # ThumbRot in [-120, 0]
    col = (intact_hand, 'thumbRot_Pos')
    if col in scaled.columns:
        scaled[col] = (scaled[col] + 60) / 60
        scaled[col] = scaled[col].clip(-1, 1)
    # Leave EMG columns unchanged!
    return scaled

def rescale_data(angles_df, intact_hand):
    series = False
    if isinstance(angles_df, pd.Series):
        series = True
        angles_df = angles_df.to_frame().T

    df = angles_df.copy()

    # 1) Finger/joint angles originally in [0…120]°
    angle_cols = [
        (intact_hand, 'index_Pos'),
        (intact_hand, 'middle_Pos'),
        (intact_hand, 'ring_Pos'),
        (intact_hand, 'pinky_Pos'),
        (intact_hand, 'thumbFlex_Pos'),
    ]
    for col in angle_cols:
        if col in df.columns:
            x = df[col]                # x ∈ [–1…+1]
            deg = 60 * x + 60          # invert (deg−60)/60 → deg = 60x + 60
            df[col] = deg.clip(0, 120)

    # 2) Thumb rotation originally in [–120…0]°
    col_tr = (intact_hand, 'thumbRot_Pos')
    if col_tr in df.columns:
        x = df[col_tr]                # x ∈ [–1…+1]
        deg = 60 * x - 60             # invert (deg+60)/60 → deg = 60x − 60
        df[col_tr] = deg.clip(-120, 0)

    if series:
        return df.iloc[0]
    else:
        return df

def load_data(data_dir, intact_hand, features, targets, perturber=None):
    """
    Load one experiment's EMG and angle data, build a combined DataFrame.
    Handles missing features or extra features gracefully.
    """
    import os
    # 1) Load angles
    angle_path = os.path.join(data_dir, 'aligned_angles.parquet')
    angles_df = pd.read_parquet(angle_path)

    # If this is the aligned_angles.parquet mode, convert any stringified tuple columns back to real tuples
    if os.path.basename(angle_path) == 'aligned_angles.parquet':
        import ast
        def _maybe_tuple(x):
            if isinstance(x, str) and x.startswith('(') and x.endswith(')'):
                try:
                    y = ast.literal_eval(x)
                    if isinstance(y, tuple):
                        return y
                except Exception:
                    pass
            return x
        angles_df.columns = pd.Index([_maybe_tuple(c) for c in angles_df.columns])

    # 2) Load EMG and timestamps
    emg_arr = np.load(os.path.join(data_dir, 'aligned_filtered_emg.npy'))  # shape (T_emg, C_emg)
    ts = np.load(os.path.join(data_dir, 'aligned_timestamps.npy'))        # shape (T_emg,)

    # 3) Determine EMG feature names - only use as many as channels present
    emg_feature_names = features[:emg_arr.shape[1]]

    # 4) Build DataFrame for EMG
    df_emg = pd.DataFrame(emg_arr, index=ts, columns=emg_feature_names)

    # 5) Build DataFrame for angles, set index to timestamp for alignment
    df_angles = angles_df.set_index('timestamp')

    # print("EMG index unique?", df_emg.index.is_unique)
    # print("Angles index unique?", df_angles.index.is_unique)
    # print("EMG index sample:", df_emg.index[:10])
    # print("Angles index sample:", df_angles.index[:10])


    # 6) Concatenate EMG and angles on the timestamp index (inner join)
    data = pd.concat([df_emg, df_angles], axis=1, join='inner')

    # print(f"→ load_data: after concat, {len(data)} rows × {len(data.columns)} columns")

    # 7) Apply perturbation if provided
    if perturber is not None:
        data = perturber.apply(data)
    
    # print("\n--- Before scaling ---")
    # print("EMG min:", data[features].min().min())
    # print("EMG max:", data[features].max().max())
    # print("Angle min:", data[targets].min().min())
    # print("Angle max:", data[targets].max().max())

    # 8) Scale data (handle missing columns inside)# 
    data[targets] = scale_data(data[targets], intact_hand)

    # print("\n--- After scaling ---")
    # print("EMG min:", data[features].min().min())
    # print("EMG max:", data[features].max().max())
    # print("Angle min:", data[targets].min().min())
    # print("Angle max:", data[targets].max().max())

    return data

# def load_data(data_dir, intact_hand, features, perturber=None):
#     angles = pd.read_parquet(join(data_dir, 'cropped_smooth_angles.parquet'))
#     angles.index = range(len(angles))
#     try:
#         emg = np.load(join(data_dir, 'cropped_aligned_emg.npy'))
#     except:
#         emg = np.load(join(data_dir, 'cropped_emg.npy'))
# 
#     data = angles.copy()
#     data = scale_data(data, intact_hand)
# 
#     int_features = [int(feature[1]) for feature in features]
#     emg = emg[:, int_features]
#     if perturber is not None:
#         emg = (perturber @ emg.T).T
# 
#     for feature in features:
#         data[tuple(feature)] = emg[:, int_features.index(int(feature[1]))]
# 
#     return data


def get_data(config, data_dirs, intact_hand, visualize=False, test_dirs=None, perturb_file=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    trainsets = []
    valsets = []
    testsets = []
    combined_sets = []

    # If a perturbation file was given, load it; otherwise perturber = None
    perturber = np.load(perturb_file) if perturb_file is not None else None

    # ─── LOOP OVER “TRAIN/VAL” DIRECTORIES ────────────────────────────────────────
    for recording_id, data_dir in enumerate(data_dirs):
        data = load_data(data_dir, intact_hand, config.features, config.targets, perturber)
        # print(f"Data length for recording {recording_id} ({data_dir}): {len(data)}")

        if visualize:
            # print("get_data debug ")
            # print("Available DataFrame columns:")
            # for col in data.columns:
            #     print("   ", col)
            # print("Requested targets (config.targets):")
            # for tgt in config.targets:
            #     print("   ", tgt)
            # missing = [t for t in config.targets if t not in data.columns]
            # print("Missing from DataFrame:", missing)

            # Plot features and targets for visual debugging
            axs = data[config.features].plot(subplots=True, ylim=(-0.1, 1.1))
            for ax in axs:
                ax.legend(loc='upper right')
            plt.suptitle(f'Features {config.recordings[recording_id]}')
            plt.show()

            axs = data[config.targets].plot(subplots=True, ylim=(-1.1, 1.1))
            for ax in axs:
                ax.legend(loc='upper right')
            plt.suptitle(f'Targets {config.recordings[recording_id]}')
            plt.show()

        # ─── SPLIT “TRAIN vs. VAL” USING .iloc ─────────────────────────────────────
        split_idx = len(data) // 5 * 4
        if split_idx <= 0 or split_idx >= len(data):
            # If data is too small to form a proper 80/20 split, put everything in train and leave val empty
            train_set = data.copy()
            val_set = pd.DataFrame(columns=data.columns)
        else:
            train_set = data.iloc[:split_idx].copy()
            val_set   = data.iloc[split_idx:].copy()

        trainsets.append(train_set)
        valsets.append(val_set)
        combined_sets.append(data.copy())

    # ─── LOOP OVER ANY “TEST” DIRECTORIES ────────────────────────────────────────
    if test_dirs is not None:
        for test_dir in test_dirs:
            data = load_data(test_dir, intact_hand, config.features, config.targets, perturber)

            split_idx = len(data) // 5 * 4
            if split_idx <= 0 or split_idx >= len(data):
                # Too few points → test_set remains empty
                test_set = pd.DataFrame(columns=data.columns)
            else:
                test_set = data.iloc[split_idx:].copy()

            testsets.append(test_set)

    return trainsets, valsets, combined_sets, testsets



def train_model(trainsets, valsets, testsets, device, wandb_mode, wandb_project, wandb_name, config=None, person_dir='test'):
    # with wandb.init(mode=wandb_mode, project=wandb_project, name=wandb_name, config=config):
    with wandb.init(mode=wandb_mode, project=wandb_project, name=wandb_name, config=config.to_dict()):
        config = wandb.config

        # weight_decay = getattr(config, "weight_decay", 0.0)
        # model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets), weight_decay=config.weight_decay,  **config)
        model = TimeSeriesRegressorWrapper(device=device, input_size=len(config.features), output_size=len(config.targets),  **config)

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

                print(f"\nStarting epoch {epoch}...")
                train_loss = model.train_one_epoch(dataloader)

                val_loss, test_loss, val_losses = evaluate_model(model, valsets, testsets, device, config)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wandb.run.summary['best_epoch'] = epoch
                    wandb.run.summary['best_val_loss'] = best_val_loss
                    # Save best model
                    model.save(join('data', person_dir, 'models', f'{wandb_name}_best.pt'))
                if test_loss < wandb.run.summary.get('best_test_loss', float('inf')):
                    wandb.run.summary['best_test_loss'] = test_loss
                    wandb.run.summary['best_test_epoch'] = epoch
                wandb.run.summary['used_epochs'] = epoch

                print(f"Epoch {epoch} validation loss: {val_loss}, test loss: {test_loss}")

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
        model.save(join('data', person_dir, 'models', f'{wandb_name}_bs{config.batch_size}_sl{config.seq_len}.pt'))
        return model


def evaluate_model(model, valsets, testsets, device, config):
    warmup_steps = config.warmup_steps  # todo
    val_losses = []
    for set_id, val_set in enumerate(valsets):
        # print(f"Evaluating val_set {set_id}: length = {len(val_set)}")
        # Print first few rows or shape to inspect data
        # print(f"val_set columns: {val_set.columns.tolist()}")
        # print(f"val_set head:\n{val_set.head()}")

        # Predict
        val_pred = model.predict(val_set, config.features, config.targets).squeeze(0)
        # print(f"val_pred shape: {val_pred.shape}")# 

        # Calculate loss
        target_tensor = torch.tensor(val_set[config.targets].values, dtype=torch.float32).to(device)
        # print(f"target_tensor shape: {target_tensor.shape}")

        loss = model.criterion(val_pred[warmup_steps:], target_tensor[warmup_steps:])
        loss = float(loss.to('cpu').detach())
        val_losses.append(loss)

    total_val_loss = sum(val_losses) / len(val_losses)

    test_losses = []
    for set_id, test_set in enumerate(testsets):
        # print(f"Evaluating test_set {set_id}: length = {len(test_set)}")
        # print(f"test_set columns: {test_set.columns.tolist()}")
        # print(f"test_set head:\n{test_set.head()}")

        test_pred = model.predict(test_set, config.features, config.targets).squeeze(0)
        # print(f"test_pred shape: {test_pred.shape}")

        target_tensor = torch.tensor(test_set[config.targets].values, dtype=torch.float32).to(device)
        # print(f"target_tensor shape: {target_tensor.shape}")

        loss = model.criterion(test_pred[warmup_steps:], target_tensor[warmup_steps:])
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
        start_idx = idx * self.seq_len + self.index_shift
        end_idx = (idx + 1) * self.seq_len + self.index_shift
        # print(f"Fetching idx: {idx}, set_idx: {set_idx}, slice: {start_idx}:{end_idx}")
        x = torch.tensor(self.data_sources[set_idx].loc[idx * self.seq_len + self.index_shift: (idx + 1) * self.seq_len + self.index_shift - 1, self.features].values, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data_sources[set_idx].loc[idx * self.seq_len + self.index_shift: (idx + 1) * self.seq_len + self.index_shift - 1, self.targets].values, dtype=torch.float32, device=self.device)
        # print(f"x shape: {x.shape}, y shape: {y.shape}")
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
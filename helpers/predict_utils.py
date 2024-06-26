from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import random
from helpers.models import TorchTimeSeriesClassifier

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
#     print('Using MPS')
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
#     print('Using CUDA')
# else:
device = torch.device("cpu")
#     print('Using CPU')


def train_model(trainsets, testsets, mode='online', config=None):
    with wandb.init(mode=mode):
        if mode != 'disabled':
            config = wandb.config
        model = TorchTimeSeriesClassifier(input_size=len(config.features), hidden_size=config.hidden_size,
                                          output_size=len(config.targets), n_epochs=config.n_epochs, seq_len=config.seq_len,
                                          learning_rate=config.learning_rate,
                                          warmup_steps=config.warmup_steps, num_layers=config.n_layers,
                                          model_type=config.model_type)
        model.to(device)

        dataset = TSDataset(trainsets, config.features, config.targets, sequence_len=125)
        dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

        best_val_loss = float('inf')
        best_epoch = 0
        epochs_no_improve = 0
        patience = 7
        used_epochs = 0
        for epoch in range(model.n_epochs):
            used_epochs = epoch
            model.train_one_epoch(dataloader)

            if epoch % 1 == 0:
                losses = []
                for set_id, test_set in enumerate(testsets):
                    val_pred = model.predict(test_set, config.features).squeeze(0)
                    loss = model.eval_criterion(val_pred[config.warmup_steps:],
                                                torch.tensor(test_set[config.targets].values, dtype=torch.float32)[
                                                config.warmup_steps:].to(device))
                    loss = float(loss.to('cpu').detach())
                    losses.append(loss)
                total_loss = sum(losses) / len(losses)
                print('Total val loss:', total_loss)
                log = {f'val_loss_{set_id}': loss for set_id, loss in enumerate(losses)}
                log['total_val_loss'] = total_loss
                wandb.log(log, step=epoch)

                if total_loss < best_val_loss:
                    best_val_loss = total_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve > patience:
                    print('Early stopping, epoch:', epoch)
                    break

        wandb.run.summary['used_epochs'] = used_epochs
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_val_loss'] = best_val_loss


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary['parameters'].items():
            for key_2, value_2 in value.items():
                if isinstance(value_2, list):
                    for id, value_3 in enumerate(value_2):
                        if isinstance(value_3, list):
                            value_2[id] = tuple(value_3)
                setattr(self, key, value_2)


class TSDataset(Dataset):
    def __init__(self, data_sources, features, targets, sequence_len, index_shift=0, dummy_labels=False, device='cpu'):
        self.data_sources = data_sources
        for i in range(len(self.data_sources)):
            self.data_sources[i] = self.data_sources[i].astype('float32')
        self.features = features
        self.targets = targets
        self.sequence_len = sequence_len
        self.index_shift = index_shift
        self.lengths = [len(data) // self.sequence_len - 1 for data in self.data_sources]
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
        x = torch.tensor(self.data_sources[set_idx].loc[idx * self.sequence_len + self.index_shift: (idx + 1) * self.sequence_len + self.index_shift - 1, self.features].values, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data_sources[set_idx].loc[idx * self.sequence_len + self.index_shift: (idx + 1) * self.sequence_len + self.index_shift - 1, self.targets].values, dtype=torch.float32, device=self.device)
        if self.dummy_labels:
            l = torch.ones_like(y, dtype=torch.float32, device=self.device)
            return x, y, l
        else:
            return x, y


    def set_index_shift(self, shift):
        self.index_shift = shift


class TSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_index_shift(random.randint(0, self.dataset.sequence_len))
        return super().__iter__()

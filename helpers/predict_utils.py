from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import random


def train_model(used_folds, config=None):
    with wandb.init(config):
        config = wandb.config

        mean_best_val_loss = 0
        mean_used_epochs = 0


        for fold in used_folds:
            dataset = TSDataset(train, features, ['y'], sequence_len=125)
            dataloader = TSDataLoader(dataset, batch_size=2, shuffle=True)
            model = TorchTimeSeriesClassifier(input_size=len(features), hidden_size=config.hidden_size, n_epochs=config.max_n_epochs,
                                              seq_len=config.seq_len, learning_rate=config.learning_rate,
                                              warmup_steps=config.warmup_steps, num_layers=config.n_layers,
                                              model_type=config.model_type)
            best_val_loss = float('inf')
            epochs_no_improve = 0
            patience = 99
            used_epochs = 0
            for epoch in range(model.n_epochs):
                used_epochs = epoch
                model.train_one_epoch(fold['train'], config.used_features)

                if epoch % 50 == 49:
                    val_pred = model.predict(fold['val'], config.used_features)
                    loss = model.eval_criterion(val_pred[config.warmup_steps:],
                                                torch.tensor(fold['val']['y'].values, dtype=torch.float32)[
                                                config.warmup_steps:]).item()
                    val_pred = val_pred.numpy()
                    val_pred = (val_pred > 0.5).astype(int)
                    accuracy = ((val_pred[config.warmup_steps:]) == fold['val']['y'].values[config.warmup_steps:]).mean()
                    sensitivity = ((val_pred[config.warmup_steps:] == 1) & (fold['val']['y'][config.warmup_steps:] == 1)).sum() / (
                                fold['val']['y'][config.warmup_steps:] == 1).sum()
                    specificity = ((val_pred[config.warmup_steps:] == 0) & (fold['val']['y'][config.warmup_steps:] == 0)).sum() / (
                                fold['val']['y'][config.warmup_steps:] == 0).sum()
                    auc = roc_auc_score(fold['val']['y'][config.warmup_steps:], val_pred[config.warmup_steps:])

                    wandb.log({'val_loss': loss, 'val_accuracy': accuracy, 'sensitivity': sensitivity,
                               'specificity': specificity, 'auc': auc})

                    if loss < best_val_loss:
                        best_val_loss = loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 50
                    if epochs_no_improve > patience:
                        print('Early stopping')
                        break

            mean_best_val_loss += best_val_loss
            mean_used_epochs += used_epochs
        mean_best_val_loss /= len(used_folds)
        mean_used_epochs /= len(used_folds)

        wandb.log({'mean_best_val_loss': mean_best_val_loss, 'used_epochs': mean_used_epochs})

def plot_auc_roc(y_true, y_scores, name):
    # Calculate AUC
    auc = roc_auc_score(y_true, y_scores)
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} ROC curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_results(target, pred, binary_pred, name):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(target)), pred, color='purple', label='Prediction')
    plt.plot(np.arange(len(target)), binary_pred, color='orange', label='Binary Prediction')
    plt.plot(np.arange(len(target)), target, color='green', label='Target')
    plt.legend()
    plt.title(f'{name} Prediction')
    plt.show()

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)



class TSDataset(Dataset):
    def __init__(self, data, features, targets, sequence_len, index_shift=0):
        self.data = data
        self.data = self.data.astype('float32')
        self.features = features
        self.targets = targets
        self.sequence_len = sequence_len
        self.index_shift = index_shift

    def __len__(self):
        return len(self.data) // self.sequence_len - 1

    def __getitem__(self, idx):
        x = self.data.loc[idx * self.sequence_len + self.index_shift: (idx + 1) * self.sequence_len+ self.index_shift - 1, self.features].values
        y = self.data.loc[idx * self.sequence_len + self.index_shift: (idx + 1) * self.sequence_len+ self.index_shift - 1, self.targets].values
        return x, y

    def set_index_shift(self, shift):
        self.index_shift = shift


class TSDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        self.dataset.set_index_shift(random.randint(0, self.dataset.sequence_len))
        return super().__iter__()

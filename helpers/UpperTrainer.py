# BUGMAN Nov 30 2021
# This trainer is used only for offline verification

import sys
import os

import torch
import torch.nn as nn
from tqdm import tqdm

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np

# torch.autograd.set_detect_anomaly(True)

class Offline_Trainer():
    def __init__(self, model, save_path, device, early_stopping=np.inf, warmup_steps=0, dt=0.016666667, clip=4, EMG_mapping_flexible=1):
        """Offline Trainer
        Args:
            save_path (string): Trained model will be saved at this location
            device : The device used to run the training
            warmup_steps (int): Number of steps to warm up the model
            dt (float, optional): Time between each data point. Defaults to 0.016666667.
            clip (int, optional): `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        """

        self.model = model
        self.save_path = save_path
        self.device = device
        self.early_stopping = early_stopping
        self.warmup_steps = warmup_steps
        self.dt = dt
        self.clip = clip
        self.EMG_mapping_flexible = EMG_mapping_flexible

        self.outLambda = 1
        self.predictionLambda = 0
        self.activationLambda = 0

        self.outLossFunc = nn.MSELoss().to(self.device)
        self.predictionLossFunc = nn.BCELoss().to(self.device)
        self.activationLossFunc = lambda x: 0


    def train_one_epoch(self, optimizer, data_loaders, epoch, scheduler=None):
        """
        Args:
            optimizer ([torch.optim]): optimizer
            data_loaders ([DataLoader]): data loader dictionary with train and test sets
            epoch ([int]): epoch number used for saving
            scheduler ([torch.optim.lr_scheduler])): scheduler for decreasing learning rate
        """

        for method in ['train', 'test']:
            print('\n*********************************\n')
            print(f'{method} epoch {epoch}...')

            dataloader = data_loaders[method]
            if method == 'train':
                self.model.train()
                torch.set_grad_enabled(True)
            lossTot = 0
            for batch_index, (EMG, target, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
                # batch[0]: EMG[batch_index, sequence_index, EMG_channel]
                # batch[1]: target[batch_index, sequence_index, Joint_Index]
                # batch[2]: label[batch_index, sequence_index, Joint_Index]

                ss = torch.zeros((target.shape[0], 2 * target.shape[1]), dtype=torch.float, device=self.device)
                outVals = torch.zeros_like(target, dtype=torch.float, device=self.device)
                preds = torch.zeros_like(labels, dtype=torch.float, device=self.device)
                activation = torch.zeros((target.shape[0], target.shape[1], target.shape[2] * 2), dtype=torch.float, device=self.device)

                for i in range(EMG.shape[1]):
                    iEMG = EMG[:, i, :]

                    w, ss, probs, alphas = self.model(ss, iEMG, self.dt)

                    outVals[:, i, :] = w
                    preds[:, i, :] = probs
                    activation[:, i, :] = alphas
                outLosses = torch.tensor([self.outLossFunc(outVals[:, self.warmup_steps:, i ], target[:, self.warmup_steps:, i ]) for i in range(target.shape[2])], dtype=torch.float, device=self.device)
                predictionLoss = self.predictionLossFunc(preds[:, self.warmup_steps:, : ], labels[:, self.warmup_steps:, : ])
                activationLoss = self.activationLossFunc(torch.abs(activation[:, self.warmup_steps:, : ]))

                loss = self.outLambda * torch.sum(outLosses) + self.predictionLambda * predictionLoss + self.activationLambda * activationLoss
                lossTot += loss.item()

                if method == 'train':
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    optimizer.step()

                if torch.isnan(loss): raise ValueError(f'{method} loss became nan at epoch {epoch}')

            averageLoss = lossTot/len(dataloader)
            print(f'\t{method} loss: {averageLoss:.6f}')

        if scheduler is not None:
            scheduler.step(averageLoss)  # todo

        print(f'Parameters at the end of epoch {epoch}:')
        self.model.print_params()
            
        return averageLoss




    def saveModel(self, epoch, optimizer, loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            self.save_path)
        print(f'Epoch {epoch} is finished. Model saved at {self.save_path} as validation loss is decreasing.')

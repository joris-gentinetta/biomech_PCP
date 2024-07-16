import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from abc import ABC, abstractmethod
from tqdm import tqdm
from bilinear import Muscles, Joints

SR = 60

class TimeSeriesRegressor(nn.Module, ABC):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    @abstractmethod
    def get_starting_states(self, batch_size, y=None):
        pass

    @abstractmethod
    def forward(self, x, states):
        pass


class RNN(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.model_type = kwargs.get('model_type')
        self.hidden_size = kwargs.get('hidden_size')
        self.n_layers = kwargs.get('n_layers')

        if self.model_type == 'RNN':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.n_layers, batch_first=True)
        else:
            raise ValueError(f'Unknown RNN type {self.model_type}')
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def get_starting_states(self, batch_size, y=None):
        if self.model_type == 'LSTM':
            states = (torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device))
        else:
            states = torch.zeros(self.n_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        return states

    def forward(self, x, states):
        out, states = self.rnn(x, states)
        out = self.fc(out)
        return out, states


class CNN(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.hidden_size = kwargs.get('hidden_size')
        self.n_layers = kwargs.get('n_layers')
        self.cnn = nn.Conv1d(self.input_size, out_channels=self.hidden_size, kernel_size=10, stride=1, padding=9)
        if self.n_layers == 2:
            self.cnn2 = nn.Conv1d(self.hidden_size, out_channels=2*self.hidden_size, kernel_size=20, stride=1, padding=19)
            self.fc = nn.Linear(2*self.hidden_size*10, self.output_size)
        else:
            self.fc = nn.Linear(self.hidden_size*10, self.output_size)

    def get_starting_states(self, batch_size, y=None):
        return None

    def forward(self, x, states=None):
        # x.shape = (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # x.shape = (batch_size, input_size, seq_len)
        out = self.cnn(x)  # out.shape = (batch_size, hidden_size, seq_len)
        if self.n_layers == 2:
            out = self.cnn2(out)
        out = out.permute(0, 2, 1)  # out.shape = (batch_size, seq_len, hidden_size)
        out = out[:, :x.shape[2], :]

        # pad to make sure that only past data is used:
        first_value = out[:, 0, :].unsqueeze(1)
        padded_first_value = first_value.repeat(1, 9, 1)
        out = torch.cat([padded_first_value, out], dim=1)

        out = torch.concat([self.fc(out[:, i:i+10, :].flatten(start_dim=1, end_dim=2)).unsqueeze(1) for i in range(x.shape[2])], dim=1)

        return out, None


class DenseNet(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        hidden_size = kwargs.get('hidden_size')
        n_layers = kwargs.get('n_layers')

        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(p=0.4))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.4))

        layers.append(nn.Linear(hidden_size, self.output_size))

        self.model = nn.Sequential(*layers)

    def get_starting_states(self, batch_size, y=None):
        return None

    def forward(self, x, states=None):
        out = self.model(x)
        return out, states


class StatefulDenseNet(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        hidden_size = kwargs.get('hidden_size')
        n_layers = kwargs.get('n_layers')

        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(p=0.4))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.4))

        layers.append(nn.Linear(hidden_size, self.output_size))

        self.model = nn.Sequential(*layers)

    def get_starting_states(self, batch_size, y=None):
        return y[:, 0:1, :]

    def forward(self, x, states=None):
        # out = self.model(x)
        # states = states + out
        # return states, states

        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            update = self.model(x[:, i:i+1, :])
            states = states + update
            out[:, i:i+1, :] = states
        return out, states

class PhysMuscleModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)

        self.output_size = output_size

        if input_size * 2 != output_size:
            raise ValueError(f'PhysMuscleModel: Input size must be 1/2 times the output size, got {input_size} and {output_size}')

        self.model = Muscles(device=device, n_joints=output_size // 4)

    def get_starting_states(self, batch_size, y=None):
        return None

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        activations = x.reshape(x.shape[0], x.shape[1], self.input_size // 2, 2)
        for i in range(x.shape[1]):
                F, K = self.model(activations[:, i, :, :], states[1])
                out[:, i, :] = torch.cat([F, K], dim=2).reshape(out.shape[0], out.shape[2])
        return out, None


class PhysJointModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)

        self.output_size = output_size

        if input_size != output_size * 4:
            raise ValueError(
                f'PhysJointModel: Input size must be 4 times the output size, got {input_size} and {output_size}')

        self.model = Joints(device=device, n_joints=output_size, dt=1 / SR, speed_mode=False)

    def get_starting_states(self, batch_size, y=None):
        theta = y[:, 0, :]
        d_theta = (y[:, 1, :] - y[:, 0, :]) * SR
        states = torch.stack([theta, d_theta], dim=2)
        return [states.unsqueeze(3) * self.model.M.unsqueeze(0).unsqueeze(2), states] # todo dimensions

    def forward(self, x, joint_states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        F, K = x.reshape(x.shape[0], x.shape[1], self.output_size, 4).split(dim=3, split_size=2)
        for i in range(x.shape[1]):
            out[:, i, :], muscle_states, joint_states = self.model(F[:, i, :], K[:, i, :], joint_states)
        return out, muscle_states, joint_states


class ModularModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        activation_model = kwargs.get('activation_model')
        muscle_model = kwargs.get('muscle_model')
        joint_model = kwargs.get('joint_model')

        self.device = device

        self.activation_model = self.get_model(input_size, output_size * 2, activation_model)
        self.sigmoid = nn.Sigmoid()
        self.muscle_model = self.get_model(output_size * 2, output_size * 4, muscle_model)
        self.joint_model = self.get_model(output_size * 4, output_size, joint_model)

    def get_model(self, input_size, output_size, config):
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
        elif config['model_type'] == 'StatefulDenseNet':
            modelclass = StatefulDenseNet
        elif config['model_type'] == 'CNN':
            modelclass = CNN
        elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
            modelclass = RNN
        elif config['model_type'] == 'PhysMuscleModel':
            modelclass = PhysMuscleModel
        elif config['model_type'] == 'PhysJointModel':
            modelclass = PhysJointModel
        else:
            raise ValueError(f'Unknown model type {config["model_type"]}')
        return modelclass(input_size, output_size, self.device, **config)


    def get_starting_states(self, batch_size, y=None):
        return [self.activation_model.get_starting_states(batch_size, y),  [self.muscle_model.get_starting_states(batch_size, y), self.joint_model.get_starting_states(batch_size, y)[0]], self.joint_model.get_starting_states(batch_size, y)[1]]

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            activation_out, states[0] = self.activation_model(x[:, i:i+1, :], states[0])
            activation_out = self.sigmoid(activation_out)
            muscle_out, states[1][0] = self.muscle_model(activation_out, states[1])
            out[:, i:i+1, :], states[1][1], states[2] = self.joint_model(muscle_out, states[2])

        return out, states



class TimeSeriesRegressorWrapper:
    def __init__(self, input_size, output_size, device, n_epochs, seq_len, learning_rate, warmup_steps, model_type, **kwargs):

        if model_type == 'DenseNet':
            self.model = DenseNet(input_size, output_size, device, **kwargs)
        elif model_type == 'StatefulDenseNet':
            self.model = StatefulDenseNet(input_size, output_size, device, **kwargs)
        elif model_type == 'CNN':
            self.model = CNN(input_size, output_size, device, **kwargs)
        elif model_type in ['RNN', 'LSTM', 'GRU']:
            self.model = RNN(input_size, output_size, device, **kwargs)
        elif model_type == 'ModularModel':
            self.model = ModularModel(input_size, output_size, device, **kwargs)
        else:
            raise ValueError(f'Unknown model type {model_type}')

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)


        self.n_epochs = n_epochs
        self.seq_len = seq_len
        self.warmup_steps = warmup_steps

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
        return self

    def train_one_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for x, y in dataloader:
            states = self.model.get_starting_states(dataloader.batch_size, y)
            outputs, states = self.model(x, states)

            loss = self.criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            # print(loss.item())
            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def predict(self, test_set, features, targets):
        self.model.eval()
        x = torch.tensor(test_set.loc[:, features].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        y = torch.tensor(test_set.loc[:, targets].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            states = self.model.get_starting_states(1, y)
            y_pred, _ = self.model(x, states=states)
        return y_pred

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self



# if __name__ == '__main__':
#     model = ModularModel(device=torch.device('cpu'), input_size=8, output_size=4, muscleType='bilinear', nn_ratio=0.2)
#     x = torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=torch.float).unsqueeze(1).repeat(1, 125, 1, 1) / 2
#     states = model.get_starting_states(5, x)
#     out, states = model(x, states) # x.shape = (batch_size, seq_len, n_joints, n_muscles_per_joint)
#     print(out)


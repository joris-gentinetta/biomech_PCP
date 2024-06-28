import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.functional import F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, model_type='LSTM'):
        super().__init__()
        self.hidden_size = hidden_size
        self.model_type = model_type
        self.num_layers = num_layers
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f'Unknown RNN type {self.model_type}')
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x, states):
        out, states = self.rnn(x, states)
        out = self.fc(out)
        return out, states


class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Conv1d(input_size, out_channels=hidden_size, kernel_size=10, stride=1, padding=9)
        if self.num_layers == 2:
            self.cnn2 = nn.Conv1d(hidden_size, out_channels=2*hidden_size, kernel_size=20, stride=1, padding=19)
            self.fc = nn.Linear(2*hidden_size*10, output_size)
        else:
            self.fc = nn.Linear(hidden_size*10, output_size)
        self.dummy_state = torch.zeros(self.num_layers, 1, 1, requires_grad=False)


    def forward(self, x, states=None):
        # x.shape = (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)  # x.shape = (batch_size, input_size, seq_len)
        out = self.cnn(x)  # out.shape = (batch_size, hidden_size, seq_len)
        if self.num_layers == 2:
            out = self.cnn2(out)
        out = out.permute(0, 2, 1)  # out.shape = (batch_size, seq_len, hidden_size)
        out = out[:, :x.shape[2], :]

        # pad to make sure that only past data is used:
        first_value = out[:, 0, :].unsqueeze(1)
        padded_first_value = first_value.repeat(1, 9, 1)
        out = torch.cat([padded_first_value, out], dim=1)

        out = torch.concat([self.fc(out[:, i:i+10, :].flatten(start_dim=1, end_dim=2)).unsqueeze(1) for i in range(x.shape[2])], dim=1)

        return out, self.dummy_state


class upperExtremityModel(nn.Module):
    def __init__(self, device, input_size, muscleType='bilinear', output_size=4):
        super().__init__()

        self.device = device
        self.output_size = output_size
        self.muscleType = muscleType

        jointModels = {'bilinear': self.bilinearInit}

        if muscleType == 'bilinear':
            from dynamics.Joint_1dof_Bilinear_NN import Joint_1dof
        else:
            raise ValueError(f'{muscleType} not implemented')

        self.numStates = None
        self.params = {}

        jointModels[muscleType]()

        self.AMIDict = []
        self.JointDict = nn.ModuleList()
        for i in range(self.output_size):
            self.AMIDict.append([self.muscleDict[i][0], self.muscleDict[i][1]])
            self.JointDict.append(Joint_1dof(self.device, self.AMIDict[i], self.params, self.lr))


    def bilinearInit(self):
        from dynamics.Muscle_bilinear import Muscle

        # muscle params
        K0 = 100
        K1 = 2000
        L0 = 0.06
        L1 = 0.006
        M = 0.05

        self.muscleDict = []
        for _ in range(self.output_size):
            self.muscleDict.append([Muscle(K0, K1, L0, L1, [-M]), Muscle(K0, K1, L0, L1, [M])])

        self.params = {'I': [0.004], 'K': 5, 'B': .3, 'K_': 5, 'B_': .3,
                       'speed_mode': False, 'K0_': 2000, 'K1_': 40000, 'L0_': 1.2, 'L1_': 0.12, 'I_': 0.064, 'M_': 0.1}
        self.numStates = 2

    def forward(self, x, states, dt):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size ), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            for j in range(self.output_size):
                out[:, i, j], states[j] = self.JointDict[i](states[:, i * self.numStates:(i + 1) * self.numStates],
                                                       x[:, 2 * i:2 * (i + 1)], dt)
        return out, states


class TimeSeriesRegressor:
    def __init__(self, input_size, hidden_size, output_size, n_epochs, seq_len, learning_rate, warmup_steps, num_layers, model_type):
        self.model_type = model_type
        self.device = torch.device("cpu")
        if self.model_type == 'CNN':
            self.model = CNN(input_size, hidden_size, output_size, num_layers)
        elif self.model_type == 'biophys':
            self.model = upperExtremityModel(device=self.device, input_size=input_size)
        else:
            self.model = RNN(input_size, hidden_size, output_size, num_layers, self.model_type)
        self.train_criterion = nn.MSELoss(reduction='mean')
        self.eval_criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
        for x, y in dataloader:
            if self.model_type == 'biophys':
                states = torch.zeros((y.shape[0], 2 * y.shape[1]), dtype=torch.float, device=self.device)
            elif self.model_type == 'LSTM':
                states = (torch.zeros(self.model.num_layers, dataloader.batch_size, self.model.hidden_size), torch.zeros(self.model.num_layers, dataloader.batch_size, self.model.hidden_size))
            else:
                states = torch.zeros(self.model.num_layers, dataloader.batch_size, self.model.hidden_size)

            outputs, states = self.model(x, states)
            loss = self.train_criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def predict(self, test_set, features):
        self.model.eval()
        x = torch.tensor(test_set.loc[:, features].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_type == 'biophys':
                states = torch.zeros((x.shape[0], 2 * x.shape[1]), dtype=torch.float, device=self.device)
            elif self.model_type == 'LSTM':
                states = (torch.zeros(self.model.num_layers, 1, self.model.hidden_size), torch.zeros(self.model.num_layers, 1, self.model.hidden_size))
            else:
                states = torch.zeros(self.model.num_layers, 1, self.model.hidden_size)

            y_pred, _ = self.model(x, states=states)
        return y_pred


    def to(self, device):
        self.device = device
        self.model.to(device)
        return self




import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
torch.autograd.set_detect_anomaly(True)
from abc import ABC, abstractmethod


class TimeSeriesRegressor(nn.Module, ABC):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    @abstractmethod
    def get_starting_states(self, batch_size):
        pass

    @abstractmethod
    def forward(self, x, states):
        pass


class RNN(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, model_type, hidden_size, num_layers):
        super().__init__(input_size, output_size, device)
        self.model_type = model_type
        self.num_layers = num_layers
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(self.input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f'Unknown RNN type {self.model_type}')
        self.fc = nn.Linear(hidden_size, self.output_size)

    def get_starting_states(self, batch_size):
        if self.model_type == 'LSTM':
            states = (torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device))
        else:
            states = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float, device=self.device)
        return states

    def forward(self, x, states):
        out, states = self.rnn(x, states)
        out = self.fc(out)
        return out, states


class CNN(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, hidden_size, num_layers):
        super().__init__(input_size, output_size, device)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cnn = nn.Conv1d(self.input_size, out_channels=hidden_size, kernel_size=10, stride=1, padding=9)
        if self.num_layers == 2:
            self.cnn2 = nn.Conv1d(hidden_size, out_channels=2*hidden_size, kernel_size=20, stride=1, padding=19)
            self.fc = nn.Linear(2*hidden_size*10, self.output_size)
        else:
            self.fc = nn.Linear(hidden_size*10, self.output_size)
        # self.dummy_state = torch.zeros(self.num_layers, 1, 1, requires_grad=False)

    def get_starting_states(self, batch_size):
        return None

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

        return out, None


class upperExtremityModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, muscleType='bilinear', nn_ratio=0.2):
        super().__init__(input_size, output_size, device)

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
            self.JointDict.append(Joint_1dof(self.device, self.AMIDict[i], self.params, nn_ratio, 1/60))

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

    def get_starting_states(self, batch_size):
        return torch.zeros((batch_size, self.numStates * self.output_size), dtype=torch.float, device=self.device)

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            for j in range(self.output_size):

                a, b = self.JointDict[j](states[:, j * self.numStates:(j + 1) * self.numStates],
                                                       x[:, i, j * 2:(j + 1) * 2])
                out[:, i, j], states[:, j * self.numStates:(j + 1) * self.numStates] = a, b
        return out, states


class TimeSeriesRegressorWrapper:
    def __init__(self, input_size, output_size, device, n_epochs, seq_len, learning_rate, warmup_steps, model_type, **kwargs):

        if model_type == 'biophys':
            self.model = upperExtremityModel(input_size, output_size, device, kwargs.get('muscleType'))
        elif model_type == 'CNN':
            self.model = CNN(input_size, output_size, device, kwargs.get('hidden_size'), kwargs.get('num_layers'))
        else:
            self.model = RNN(input_size, output_size, device, model_type, kwargs.get('hidden_size'), kwargs.get('num_layers'))

        self.criterion = nn.MSELoss(reduction='mean')
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
            states = self.model.get_starting_states(dataloader.batch_size)
            outputs, states = self.model(x, states)
            loss = self.criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, test_set, features):
        self.model.eval()
        x = torch.tensor(test_set.loc[:, features].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            states = self.model.get_starting_states(1)
            y_pred, _ = self.model(x, states=states)
        return y_pred

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self




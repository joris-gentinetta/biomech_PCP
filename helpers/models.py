import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from abc import ABC, abstractmethod
from tqdm import tqdm

SR = 60

class TimeSeriesRegressor(nn.Module, ABC):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    @abstractmethod
    def get_starting_states(self, batch_size, x=None):
        pass

    @abstractmethod
    def forward(self, x, states):
        pass


class RNN(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, model_type, hidden_size, n_layers, **kwargs):
        super().__init__(input_size, output_size, device)
        self.model_type = model_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        if self.model_type == 'RNN':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, n_layers, batch_first=True)
        elif self.model_type == 'LSTM':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, n_layers, batch_first=True)
        elif self.model_type == 'GRU':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, n_layers, batch_first=True)
        else:
            raise ValueError(f'Unknown RNN type {self.model_type}')
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def get_starting_states(self, batch_size, x=None):
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
    def __init__(self, input_size, output_size, device, hidden_size, n_layers, **kwargs):
        super().__init__(input_size, output_size, device)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn = nn.Conv1d(self.input_size, out_channels=hidden_size, kernel_size=10, stride=1, padding=9)
        if self.n_layers == 2:
            self.cnn2 = nn.Conv1d(hidden_size, out_channels=2*hidden_size, kernel_size=20, stride=1, padding=19)
            self.fc = nn.Linear(2*hidden_size*10, self.output_size)
        else:
            self.fc = nn.Linear(hidden_size*10, self.output_size)
        # self.dummy_state = torch.zeros(self.n_layers, 1, 1, requires_grad=False)

    def get_starting_states(self, batch_size, x=None):
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
    def __init__(self, input_size, output_size, device, hidden_size, n_layers, **kwargs):
        super().__init__(input_size, output_size, device)
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

    def get_starting_states(self, batch_size, x=None):
        return None

    def forward(self, x, states=None):
        out = self.model(x)
        return out, states

class PhysMuscle(nn.Module):


class upperExtremityModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, muscleType='bilinear', nn_ratio=0.2, **kwargs):
        super().__init__(input_size, output_size, device)

        if input_size != output_size * 2:
            raise ValueError(f'UpperExtremityModel: Input size must be 2 times the output size, got {input_size} and {output_size}')

        self.muscleType = muscleType

        jointModels = {'bilinear': self.bilinearInit}

        if muscleType == 'bilinear':
            # from dynamics.Joint_1dof_Bilinear_NN import Joint_1dof
            from dynamics.BilinearJoints import BilinearJoints
        else:
            raise ValueError(f'{muscleType} not implemented')

        self.numStates = None
        self.params = {}

        jointModels[muscleType]()

        self.AMIDict = []
        for i in range(self.output_size):
            self.AMIDict.append([self.muscleDict[i][0], self.muscleDict[i][1]])

        self.joints = BilinearJoints(self.device, self.AMIDict, self.params, 1/SR, False)



    def bilinearInit(self):
        from dynamics.Muscle_bilinear import Muscle

        # muscle params
        K0 = math.log(100)
        K1 = math.log(2000)
        L0 = math.log(0.06)
        L1 = math.log(0.006)
        M = 0.05

        self.muscleDict = []
        for _ in range(self.output_size):
            self.muscleDict.append([Muscle(K0, K1, L0, L1, -M), Muscle(K0, K1, L0, L1, M)])

        self.params = {'I': math.log(0.004), 'K': math.log(5), 'B': math.log(.3)}
        self.numStates = 2


    def get_starting_states(self, batch_size, x=None):
        theta = x[:, 0, :]
        d_theta = (x[:, 1, :] - x[:, 0, :]) * SR
        return torch.stack([theta, d_theta], dim=2)
        # return torch.zeros((batch_size, self.output_size, self.numStates), dtype=torch.float, device=self.device, requires_grad=False)

    def forward(self, x, states, forces=None):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        x = x.reshape(x.shape[0], x.shape[1], self.output_size, 2)
        for i in range(x.shape[1]):
                out[:, i, :], states, forces = self.joints(states, x[:, i, :, :], forces)
        return out, states, forces


class ModularModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, config, **kwargs):
        super().__init__(input_size, output_size, device)

        next_input_size = input_size
        if next_input_size != config['activation_model']['input_size']:
            raise ValueError(f'Activation model input size must be the same as ModularModel input size. Got {config["activation_model"]["input_size"]} and {input_size}')

        self.activation_model, next_input_size = self.get_model(config['activation_model'])
        if next_input_size != config['muscle_model']['input_size']:
            raise ValueError(f'Muscle model input size must be the same as Activation model output size. Got {config["muscle_model"]["input_size"]} and {next_input_size}')

        self.sigmoid = nn.Sigmoid()

        self.muscle_model, next_input_size = self.get_model(config['muscle_model'])
        if next_input_size != config['joint_model']['input_size']:
            raise ValueError(f'Joint model input size must be the same as Muscle model output size. Got {config["joint_model"]["input_size"]} and {next_input_size}')

        self.joint_model, next_input_size = self.get_model(config['joint_model'])
        if next_input_size != output_size:
            raise ValueError(f'Joint model output size must be the same as ModularModel output size. Got {next_input_size} and {output_size}')




    def get_model(self, config):
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
        elif config['model_type'] == 'CNN':
            modelclass = CNN
        elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
            modelclass = RNN
        else:
            raise ValueError(f'Unknown model type {config["model_type"]}')
        return modelclass(config['input_size'], config['output_size'], self.device, config['hidden_size'], config['n_layers']), config['output_size']


    def get_starting_states(self, batch_size, x=None):
        return [self.activation_model.get_starting_states(batch_size, x), self.muscle_model.get_starting_states(batch_size, x), self.joint_model.get_starting_states(batch_size, x, self.muscle_model)]

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            activation_out, states[0] = self.activation_model(x[:, i:i+1, :], states[0])
            activation_out = self.sigmoid(activation_out)
            F, K = self.muscle_model(activation_out, states[1])
            out[:, i:i+1, :], states[1], states[2] = self.joint_model(states[2], F, K)

        return out, states



class TimeSeriesRegressorWrapper:
    def __init__(self, input_size, output_size, device, n_epochs, seq_len, learning_rate, warmup_steps, model_type, **kwargs):

        if model_type == 'biophys':
            self.model = upperExtremityModel(input_size, output_size, device, kwargs.get('muscleType'))
        elif model_type == 'DenseNet':
            self.model = DenseNet(input_size, output_size, device, kwargs.get('hidden_size'), kwargs.get('n_layers'))
        elif model_type == 'CNN':
            self.model = CNN(input_size, output_size, device, kwargs.get('hidden_size'), kwargs.get('n_layers'))
        elif model_type in ['RNN', 'LSTM', 'GRU']:
            self.model = RNN(input_size, output_size, device, model_type, kwargs.get('hidden_size'), kwargs.get('n_layers'))
        elif model_type == 'ActivationAndBiophys':
            self.model = ActivationAndBiophysModel(input_size, output_size, device, kwargs.get('activation_config'), kwargs.get('biophys_config'))
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
            states = self.model.get_starting_states(dataloader.batch_size, x)
            outputs, states = self.model(x, states)

            loss = self.criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            # print(loss.item())
            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def predict(self, test_set, features):
        self.model.eval()
        x = torch.tensor(test_set.loc[:, features].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            states = self.model.get_starting_states(1, x)
            y_pred, _ = self.model(x, states=states)
        return y_pred

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self



if __name__ == '__main__':
    model = upperExtremityModel(device=torch.device('cpu'), input_size=8, output_size=4, muscleType='bilinear', nn_ratio=0.2)
    x = torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=torch.float).unsqueeze(1).repeat(1, 125, 1, 1) / 2
    states = model.get_starting_states(5, x)
    out, states = model(x, states) # x.shape = (batch_size, seq_len, n_joints, n_muscles_per_joint)
    print(out)


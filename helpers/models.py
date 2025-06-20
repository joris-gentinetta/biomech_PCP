import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch import Tensor
from typing import Optional, Tuple, List
import numpy as np


# torch.autograd.set_detect_anomaly(True)
from abc import ABC, abstractmethod
from tqdm import tqdm
from bilinear import Muscles, Joints, M as bilinear_M

from omen.realtime import RealtimeOMEN
# from omen.postprocessing import inverse_transform_predictions

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
        self.state_mode = kwargs.get('state_mode', None)

        if self.state_mode == 'stateAware':
            self.input_size += output_size

        self.tanh = nn.Tanh() if kwargs.get('tanh', False) else None

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

        if self.state_mode == 'stateful' or self.state_mode == 'stateAware':
            return [states, y[:, 0:1, :]]
        else:
            return states

    def forward(self, x, states):
        if self.state_mode == 'stateful':
            out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
            for i in range(x.shape[1]):
                update, states[0] = self.rnn(x[:, i:i + 1, :], states[0])
                update = self.fc(update)
                states[1] = states[1] + update
                out[:, i:i + 1, :] = states[1]

        elif self.state_mode == 'stateAware':
            out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
            for i in range(x.shape[1]):
                pred, states[0] = self.rnn(torch.cat((x[:, i:i + 1, :], states[1]), dim=2), states[0])
                out[:, i:i + 1, :] = self.fc(pred)

        else:
            out, states = self.rnn(x, states)
            out = self.fc(out)
            if self.tanh is not None:
                out = self.tanh(out)

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

        self.state_mode = kwargs.get('state_mode', None)
        if self.state_mode == 'stateAware':
            self.input_size += output_size

        self.tanh = nn.Tanh() if kwargs.get('tanh', False) else None

        layers = []
        layers.append(nn.Linear(self.input_size, hidden_size))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(p=0.2))

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.2))

        layers.append(nn.Linear(hidden_size, self.output_size))

        self.model = nn.Sequential(*layers)

    def get_starting_states(self, batch_size, y=None):
        if self.state_mode == 'stateful' or self.state_mode == 'stateAware':
            return y[:, 0:1, :]
        else:
            return None

    def forward(self, x, states=None):
        if self.state_mode == 'stateful':
            out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
            for i in range(x.shape[1]):
                update = self.model(x[:, i:i+1, :])
                states = states + update
                if self.tanh is not None:
                    states = self.tanh(states)
                out[:, i:i+1, :] = states

        elif self.state_mode == 'stateAware':
            out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
            for i in range(x.shape[1]):
                states = self.model(torch.cat((x[:, i:i+1, :], states), dim=2))
                if self.tanh is not None:
                    states = self.tanh(states)
                out[:, i:i+1, :] = states

        else:
            out = self.model(x)
            if self.tanh is not None:
                out = self.tanh(out)

        return out, states

class ParallelDenseNet(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.hidden_size = kwargs.get('hidden_size')
        self.n_layers = kwargs.get('n_layers')
        self.num_states = kwargs.get('num_states')

        # we should have that the input size is the number of parallel networks, with the states input stacked with each input

        # Weight tensors for multiple layers (n_layers, n_states, out_dim)
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(self.input_size, self.hidden_size if i > 0 else self.num_states,
                                     self.output_size if i == self.n_layers - 1 else self.hidden_size))
            for i in range(self.n_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.randn(self.input_size, self.output_size if i == self.n_layers - 1 else self.hidden_size))
            for i in range(self.n_layers)
        ])

        self.activation = nn.ReLU()

    def forward(self, x, states):
        """
        x: (time_points, batch, input_size)
        states: (batch, num_states) [these are updated in each iteration, I guess]
        Returns: (time_points, batch, output_size)
        """

        # First, stack x and states for the parallel networks - they should zip together
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)

        for i in range(x.shape[0]):
            for j in range(self.n_layers):
                x = torch.einsum("bms,msh->bmh", x, self.weights[i]) + self.biases[i]
                x = self.activation(x)

            # then flatten for the output - should be (batch, musc * outputs)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        return out, states

class IdealModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.placeholder = nn.Parameter(torch.ones(1, device=device))

    def get_starting_states(self, batch_size, y=None):
        return y

    def forward(self, x, states=None):
        return states * self.placeholder, states

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


class NNMuscleModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.model_type = kwargs.get('model_type')
        self.model = self.get_model(input_size + output_size, output_size, kwargs)

    def get_starting_states(self, batch_size, y=None):
        return self.model.get_starting_states(batch_size, y)

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        activations = x.reshape(x.shape[0], x.shape[1], self.input_size // 2, 2)
        for i in range(x.shape[1]):
            F, K = self.model(activations[:, i, :, :], states[1])
            out[:, i, :] = torch.cat([F, K], dim=2).reshape(out.shape[0], out.shape[2])
        return out, None

        # return self.model(x, states[0])
        # out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)  # todo size of states to calc input size
        # for i in range(x.shape[1]):
        #     model_input = torch.cat([x[:, i, :], states[1].flatten(start_dim=1)], dim=1).unsqueeze(1)  # todo torch.zeros_like for naive
        #     out[:, i:i+1, :], states[0] = self.model(model_input, states[0])
        # return out, None

    def get_model(self, input_size, output_size, config):
        if config['model_type'] == 'ParallelDenseNet':
            modelclass = ParallelDenseNet
        # elif config['model_type'] == 'ParallelGRU':
        #     modelclass = ParallelGRU
        # elif config['model_type'] == 'ParallelLSTM':
        #     modelclass = ParallelLSTM
        # elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
        #     modelclass = RNN
        else:
            raise ValueError(f'Unknown model type for NNMuscleModel {config["model_type"]}')
        return modelclass(input_size, output_size, self.device, **config)


class PhysJointModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)

        self.output_size = output_size
        speed_mode = kwargs.get('speed_mode', False)  # todo pass from further up

        if input_size != output_size * 4:
            raise ValueError(
                f'PhysJointModel: Input size must be 4 times the output size, got {input_size} and {output_size}')

        self.model = Joints(device=device, n_joints=output_size, dt=1 / SR, speed_mode=speed_mode)

    def get_starting_states(self, batch_size, y=None):
        """ these are the muscle states! """
        theta = y[:, 0, :]
        d_theta = (y[:, 1, :] - y[:, 0, :]) * SR
        states = torch.stack([theta, d_theta], dim=2)
        return [states.unsqueeze(3) * self.model.M.unsqueeze(0).unsqueeze(2), states, theta]  # todo dimensions

    def forward(self, x, joint_states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        F, K = x.reshape(x.shape[0], x.shape[1], self.output_size, 4).split(dim=3, split_size=2)
        for i in range(x.shape[1]):
            out[:, i, :], joint_states[0], joint_states[1] = self.model(F[:, i, :], K[:, i, :], joint_states[1])
            joint_states[2] = out[:, i, :]
        return out, joint_states
#
# class NNJointModel(TimeSeriesRegressor): # this is the implicit M model
#     def __init__(self, input_size, output_size, device, **kwargs):
#         super().__init__(input_size, output_size, device)
#         self.model_type = kwargs.get('model_type')
#         self.model = self.get_model(input_size, output_size * 5, kwargs)
#         self.tanh = nn.Tanh()
#
#
#     def get_starting_states(self, batch_size, y=None):  # todo make starting state a parameter?
#         theta = y[:, 0, :]
#         d_theta = (y[:, 1, :] - y[:, 0, :]) * SR
#         states = torch.stack([theta, d_theta], dim=2)
#         M = torch.ones((1, self.output_size, 1, 2))
#         M[:, :, :, 0] = -bilinear_M
#         M[:, :, :, 1] = bilinear_M
#
#         return [states.unsqueeze(3) * M,  self.model.get_starting_states(batch_size, y)]
#
#     def forward(self, x, joint_states):
#         out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
#         for i in range(x.shape[1]):
#             model_output, joint_states = self.model(x[:, i, :], joint_states)
#             out[:, i, :], muscle_states = model_output.split(dim=1, split_size=[self.output_size, model_output.shape[1] - self.output_size])
#             muscle_states = self.tanh(muscle_states.reshape(muscle_states.shape[0], self.output_size, 2, 2))
#         return out, muscle_states, joint_states
#
#     def get_model(self, input_size, output_size, config): # todo add to TimeSeriesRegressor ?
#         if config['model_type'] == 'DenseNet':
#             modelclass = DenseNet
#         elif config['model_type'] == 'CNN':
#             modelclass = CNN
#         elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
#             modelclass = RNN
#         else:
#             raise ValueError(f'Unknown model type for NNJointModel {config["model_type"]}')
#         return modelclass(input_size, output_size, self.device, **config)


class NNJointModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.model_type = kwargs.get('model_type')
        self.model = self.get_model(input_size, output_size, kwargs)
        self.tanh = nn.Tanh()

        moment_arms = torch.ones((self.output_size, 2), dtype=torch.float, device=self.device) * bilinear_M
        moment_arms[:, 0] = moment_arms[:, 0] * -1  # note that sign flipped
        self.M = nn.Parameter(data=moment_arms)

    def get_starting_states(self, batch_size, y=None):  # todo make starting state a parameter?
        theta = y[:, 0, :]
        d_theta = (y[:, 1, :] - y[:, 0, :]) * SR
        states = torch.stack([theta, d_theta], dim=2)

        return [states.unsqueeze(3) * self.M.unsqueeze(0).unsqueeze(2), self.model.get_starting_states(batch_size, y), theta]

    def forward(self, x, joint_states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            out[:, i:i+1, :], joint_states[1] = self.model(x[:, i:i+1, :], joint_states[1])
            theta = out[:, i, :]
            d_theta = (theta - joint_states[2]) * SR
            all_states = torch.stack([theta, d_theta], dim=2)
            joint_states[0] = torch.tanh(all_states.unsqueeze(3) * self.M.unsqueeze(0).unsqueeze(2))

            joint_states[2] = theta

        return out, joint_states


    def get_model(self, input_size, output_size, config):  # todo add to TimeSeriesRegressor ?
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
        elif config['model_type'] == 'CNN':
            modelclass = CNN
        elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
            modelclass = RNN
        else:
            raise ValueError(f'Unknown model type for NNJointModel {config["model_type"]}')
        return modelclass(input_size, output_size, self.device, **config)


class ModularModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.activation_model_config = kwargs.get('activation_model')
        self.muscle_model_config = kwargs.get('muscle_model')
        self.joint_model_config = kwargs.get('joint_model')

        self.device = device

        self.activation_model = self.get_model(input_size, output_size * 2, self.activation_model_config)
        self.sigmoid = nn.Sigmoid()
        self.muscle_model = self.get_model(output_size * 2, output_size * 4, self.muscle_model_config)
        self.joint_model = self.get_model(output_size * 4, output_size, self.joint_model_config)
        # self.muscle_model = PhysMuscleModel(output_size * 2, output_size * 4, self.device, **self.muscle_model_config) if self.muscle_model_config['model_type'] == 'PhysMuscleModel' else NNMuscleModel(output_size * 2, output_size * 4, self.device, **self.muscle_model_config)
        # self.joint_model = PhysJointModel(output_size * 4, output_size, self.device, **self.joint_model_config) if self.joint_model_config['model_type'] == 'PhysJointModel' else NNJointModel(output_size * 4, output_size, self.device, **self.joint_model_config)

    def get_model(self, input_size, output_size, config):
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
        elif config['model_type'] == 'CNN':
            modelclass = CNN
        elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
            modelclass = RNN
        elif config['model_type'] == 'PhysMuscleModel':
            modelclass = PhysMuscleModel
        elif config['model_type'] == 'NNMuscleModel':
            modelclass = NNMuscleModel
        elif config['model_type'] == 'PhysJointModel':
            modelclass = PhysJointModel
        elif config['model_type'] == 'NNJointModel':
            modelclass = NNJointModel
        else:
            raise ValueError(f'Unknown model type {config["model_type"]}')
        return modelclass(input_size, output_size, self.device, **config)

    def get_starting_states(self, batch_size, y=None):
        return [self.activation_model.get_starting_states(batch_size, y),  self.muscle_model.get_starting_states(batch_size, y), self.joint_model.get_starting_states(batch_size, y)]

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            activation_out, states[0] = self.activation_model(x[:, i:i+1, :], states[0])
            activation_out = self.sigmoid(activation_out)
            muscle_out, states[1] = self.muscle_model(activation_out, [states[1], states[2][0]])
            out[:, i:i+1, :], states[2] = self.joint_model(muscle_out, states[2])

        return out, states

    def forwardInference(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            activation_out, states[0] = self.activation_model(x[:, i:i+1, :], states[0])
            activation_out = self.sigmoid(activation_out)
            muscle_out, states[1] = self.muscle_model(activation_out, [states[1], states[2][0]])
            out[:, i:i+1, :], states[2] = self.joint_model(muscle_out, states[2])

        return out, states, activation_out, muscle_out, states # return all intermediates!


class CompensationModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        activation_model = kwargs.get('activation_model')
        compensation_model = kwargs.get('compensation_model')
        muscle_model = kwargs.get('muscle_model')
        joint_model = kwargs.get('joint_model')

        self.device = device

        self.activation_model = self.get_model(input_size, output_size * 2, activation_model)
        self.sigmoid = nn.Sigmoid()
        self.muscle_model = self.get_model(output_size * 2, output_size * 4, muscle_model)
        self.joint_model = self.get_model(output_size * 4, output_size, joint_model)
        self.compensation_model = self.get_model(3, 2, compensation_model) # want 3 inputs per muscle and 2 output per muscle (a force scaler and a stiffness scaler)
        self.tanh = nn.Tanh()

    def get_model(self, input_size, output_size, config):
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
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


    def get_starting_states(self, batch_size, x=None):
        return [self.activation_model.get_starting_states(batch_size, x),  [self.muscle_model.get_starting_states(batch_size, x), self.joint_model.get_starting_states(batch_size, x)[0]], self.compensation_model.get_starting_states(batch_size, x), self.joint_model.get_starting_states(batch_size, x)[1]]

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        for i in range(x.shape[1]):
            activation_out, states[0] = self.activation_model(x[:, i:i + 1, :], states[0])
            activation_out = self.sigmoid(activation_out)
            muscle_out, states[1][0] = self.muscle_model(activation_out, states[1])
            # need to do some ~rearranging~ on the muscle states here to get them as desired - states[1][1] is the muscle state
            # with dimension [batch_size, n_joints, state_num (2), muscle_num (2)]
            # we can pull out the individual muscle states, then stack along the last dimension as [act, len, vel]
            # unsqueeze along the sequence length dimension, and pass in the input as desired
            # hill_muscle_states.shape = [batch_size, seq_len, n_networks, input_size]
            hill_muscle_states = torch.concatenate([activation_out.unsqueeze(3)] + [torch.flatten(states[1][1][:, :, st, :], start_dim=1, end_dim=2).unsqueeze(2).unsqueeze(1) for st in range(2)], dim=3)
            compensation_out, states[2] = self.compensation_model(hill_muscle_states) # [batch_size, seq_len, n_networks, output_size]
            compensation_out = self.tanh(compensation_out)
            muscle_out = muscle_out * (1 + compensation_out)

            out[:, i:i+1, :], states[1][1], states[3] = self.joint_model(muscle_out, states[3])

        return out, states

# ---------------------------------------------------------------------------- #
#                                     OMEN                                     #
# ---------------------------------------------------------------------------- #


"""
Instructions:
./omen_cli.py can be used to train OMEN on data from a session. 
Ater training, OMEN gets saved to a folder (e.g. 'P_149_Left') with
the model, scalers, trained keys, and config.

Changes to TimeSeriesRegressorWrapper:
- in __init__, handle model_type == 'OMEN'
- in load, handle OMEN's loading logic.
"""

class OMENModel(TimeSeriesRegressor):
    def __init__(self,  input_size:int, output_size:int, device:torch.device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.input_size = input_size
        self.output_size = output_size
        self.device = device

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def to(self, device:torch.device):
        self.device = device
        self.model.to(device)

    def load(self, model_path:str):
        self.model = RealtimeOMEN.load(model_path)
        self.key = self.model._config.sessions[0]
        # self.x_scaler = self.model.scalers[f'X_{self.key}']
        # self.y_scaler = self.model.scalers[f'Y']
        # self.D = len(self.model.cans)

    def get_starting_states(self, batch_size:int, y:Optional[Tensor]=None) -> None:
        """
            Reset CANs and initialize buffer      
        """
        self.model.reset(self.input_size)
        return None



    def forward(self, x:Tensor, states:Optional[Tensor]=None) -> Tuple[Tensor, Optional[Tensor]]:
        """
            Forward pass for OMEN.
            It uses Portent.encoder to encode the input, 
            then loops over each CANs and unfolds its dynamics.
            It applies scaling as needed and returns the output.

            x is a tensor of inputs of shape (batch_size, seq_len, n_inputs).
            At inference time, batch_size == seq_len == 1.
            Since we don't use OMENModel for training, this function only works at inference time.
        """
        B, S, N = x.shape
        assert B == 1, f"Batch size must be 1: {B}"
        assert S == 1, f"Sequence length must be 1: {S}"
        assert N == self.input_size, f"Input size must match: {N} != {self.input_size}"
        assert self.model.is_reset, "Model must be reset before forward pass"

        # call OMEN
        out = self.model(x.squeeze()).unsqueeze(0)  # (1, n_outputs)
        return out, states
            



# ---------------------------------------------------------------------------- #
#                             TIME SERIES REGRESSOR                            #
# ---------------------------------------------------------------------------- #

class TimeSeriesRegressorWrapper:
    def __init__(self, input_size, output_size, device, n_epochs, learning_rate, warmup_steps, model_type, **kwargs):
        # ! CHANGES: added OMEN model type
        kwargs.update({'model_type': model_type})

        if model_type == 'DenseNet':
            self.model = DenseNet(input_size, output_size, device, **kwargs)
        elif model_type == 'CNN':
            self.model = CNN(input_size, output_size, device, **kwargs)
        elif model_type in ['RNN', 'LSTM', 'GRU']:
            self.model = RNN(input_size, output_size, device, **kwargs)
        elif model_type == 'ModularModel':
            self.model = ModularModel(input_size, output_size, device, **kwargs)
        elif model_type == 'IdealModel':
            self.model = IdealModel(input_size, output_size, device, **kwargs)
        elif model_type == 'CompensationModel':
            self.model = CompensationModel(input_size, output_size, device, **kwargs)
        elif model_type == 'OMEN':
            # ? CHANGE
            self.model = OMENModel(input_size, output_size, device, **kwargs)
        else:
            raise ValueError(f'Unknown model type {model_type}')
        self.model_type = model_type

        if model_type != 'OMEN':
            self.criterion = nn.MSELoss(reduction='mean')
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            # self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, amsgrad=True)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, threshold_mode='rel', threshold=0.01)

        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps # todo
        # self.warmup_steps = kwargs.get('seq_len') - 1

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        # ! CHANGES: handle OMEN's loading logic.
        if self.model_type == 'OMEN':  # ? CHANGE
            self.model.load(path)
        else:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # note that we load to cpu
        return self

    def train(self):
        self.model.train()
        return

    def eval(self):
        self.model.eval()
        return

    def train_one_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0
        for x, y in dataloader:
            states = self.model.get_starting_states(dataloader.batch_size, y)
            outputs, states = self.model(x, states)

            loss = self.criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
            self.optimizer.step()
            epoch_loss += loss

            if torch.any(torch.isnan(loss)):
                print('NAN Loss!')

        return epoch_loss.item() / len(dataloader)

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

    def eval(self):
        self.model.eval()
        return self

    # extra wrappers to expose arguments as needed
    def named_buffers(self, *args, **kwargs):
        return self.model.named_buffers(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def modules(self, *args, **kwargs):
        return self.model.modules(*args, **kwargs)

# if __name__ == '__main__':
#     model = ModularModel(device=torch.device('cpu'), input_size=8, output_size=4, muscleType='bilinear', nn_ratio=0.2)
#     x = torch.tensor([[[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]], [[1, 2], [1, 2], [1, 2], [1, 2]]], dtype=torch.float).unsqueeze(1).repeat(1, 125, 1, 1) / 2
#     states = model.get_starting_states(5, x)
#     out, states = model(x, states) # x.shape = (batch_size, seq_len, n_joints, n_muscles_per_joint)
#     print(out)


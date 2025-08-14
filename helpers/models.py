import math
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim

# torch.autograd.set_detect_anomaly(True)
from abc import ABC, abstractmethod
from tqdm import tqdm
from dynamics.bilinear import Muscles, Joints, M as bilinear_M
#from predict_utils import simplified_enhanced_loss, simplified_muscle_physiology_loss


SR = 60

def simplified_enhanced_loss(outputs, targets, muscle_activations, force_data, warmup_steps):
    """
    Simplified combined loss for interaction model training (no material classification)
    
    Args:
        outputs: (batch, seq_len, output_size) - predicted joint positions
        targets: (batch, seq_len, output_size) - target joint positions  
        muscle_activations: (batch, seq_len, 12) - predicted muscle activations
        force_data: (batch, seq_len, 5) - force feedback data
        warmup_steps: int - steps to skip at beginning
    """
    # 1. Standard position loss
    position_loss = torch.nn.functional.mse_loss(
        outputs[:, warmup_steps:], 
        targets[:, warmup_steps:]
    )
    
    # 2. Muscle physiology loss (only after warmup)
    if muscle_activations is not None and force_data is not None:
        physiology_loss = simplified_muscle_physiology_loss(
            muscle_activations[:, warmup_steps:], 
            force_data[:, warmup_steps:],
            outputs.device
        )
        
        # Combine losses with weighting
        total_loss = position_loss + 0.3 * physiology_loss

        # print("outputs", outputs[:, warmup_steps:].abs().mean().item())
        # print("targets", targets[:, warmup_steps:].abs().mean().item())
        # print("muscle_activations", muscle_activations[:, warmup_steps:].abs().mean().item())
        # print("force_data", force_data[:, warmup_steps:].abs().mean().item())
        
        return total_loss, position_loss.item(), physiology_loss.item()
    else:
        return position_loss, position_loss.item(), 0.0

def simplified_muscle_physiology_loss(muscle_activations, force_feedback, device):
    """
    Simplified physiologically informed muscle activation loss (no material classification)
    
    Args:
        muscle_activations: (batch, seq_len, 12) - muscle activation values
        force_feedback: (batch, seq_len, 5) - force values per finger
        device: torch device
    """
    # Force magnitude across all fingers
    force_magnitude = torch.norm(force_feedback, dim=-1)  # (batch, seq_len)
    
    # 1. Activation Level Loss
    # Expected activation should scale with force magnitude
    expected_activation_level = torch.sigmoid(force_magnitude - 0.5)  # Threshold at 0.5
    actual_activation_level = torch.mean(muscle_activations, dim=-1)   # Average across all muscles
    
    activation_level_loss = torch.nn.functional.mse_loss(actual_activation_level, expected_activation_level) * 0.2
    
    # 2. Co-contraction Loss
    # During contact, antagonist muscles should co-contract for stability
    cocontraction_loss = torch.tensor(0.0, device=device)
    
    if muscle_activations.shape[-1] >= 12:  # Ensure we have enough muscle activations
        # Reshape to (batch, seq_len, 6_DOF, 2_muscles_per_DOF)
        muscle_pairs = muscle_activations.reshape(
            muscle_activations.shape[0], muscle_activations.shape[1], 6, 2
        )
        
        # Contact mask: where force is significant
        contact_mask = (force_magnitude > 0.5).float()  # (batch, seq_len)
        
        for pair_idx in range(6):
            agonist = muscle_pairs[:, :, pair_idx, 0]      # (batch, seq_len)
            antagonist = muscle_pairs[:, :, pair_idx, 1]   # (batch, seq_len)
            
            # Co-contraction: minimum activation of both muscles
            both_active = torch.minimum(agonist, antagonist)
            
            # Only penalize/reward co-contraction during contact
            cocontraction_loss += torch.mean(both_active * contact_mask) * 0.1
    
    return activation_level_loss + cocontraction_loss

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
                if torch.isnan(update).any() or torch.isinf(update).any():
                    print(f"→ [RNN.forward] NaN/Inf in RNN output at time step {i}")
                update = self.fc(update)
                if torch.isnan(update).any() or torch.isinf(update).any():
                    print(f"→ [RNN.forward] NaN/Inf after fc() at time step {i}")
                states[1] = states[1] + update
                if torch.isnan(states[1]).any() or torch.isinf(states[1]).any():
                    print(f"→ [RNN.forward] NaN/Inf in accumulated state at time step {i}")
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


# class DenseNet(TimeSeriesRegressor):
#     def __init__(self, input_size, output_size, device, **kwargs):
#         super().__init__(input_size, output_size, device)
#         hidden_size = kwargs.get('hidden_size')
#         n_layers = kwargs.get('n_layers')
# 
#         self.state_mode = kwargs.get('state_mode', None)
#         if self.state_mode == 'stateAware':
#             self.input_size += output_size
# 
#         self.tanh = nn.Tanh() if kwargs.get('tanh', False) else None
# 
#         layers = []
#         layers.append(nn.Linear(self.input_size, hidden_size))
#         layers.append(nn.LeakyReLU())
#         layers.append(nn.Dropout(p=0.2))
# 
#         for _ in range(n_layers - 1):
#             layers.append(nn.Linear(hidden_size, hidden_size))
#             layers.append(nn.LeakyReLU())
#             layers.append(nn.Dropout(p=0.2))
# 
#         layers.append(nn.Linear(hidden_size, self.output_size))
# 
#         self.model = nn.Sequential(*layers)
# 
#     def get_starting_states(self, batch_size, y=None):
#         if self.state_mode == 'stateful' or self.state_mode == 'stateAware':
#             return y[:, 0:1, :]
#         else:
#             return None
# 
#     def forward(self, x, states=None):
#         if self.state_mode == 'stateful':
#             out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
#             for i in range(x.shape[1]):
#                 update = self.model(x[:, i:i+1, :])
#                 states = states + update
#                 if self.tanh is not None:
#                     states = self.tanh(states)
#                 out[:, i:i+1, :] = states
# 
#         elif self.state_mode == 'stateAware':
#             out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
#             for i in range(x.shape[1]):
#                 states = self.model(torch.cat((x[:, i:i+1, :], states), dim=2))
#                 if self.tanh is not None:
#                     states = self.tanh(states)
#                 out[:, i:i+1, :] = states
# 
#         else:
#             out = self.model(x)
#             if self.tanh is not None:
#                 out = self.tanh(out)
# 
#         return out, states
class DenseNet(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        hidden_size = kwargs.get('hidden_size')
        n_layers    = kwargs.get('n_layers')

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
        if self.state_mode in ['stateful', 'stateAware']:
            return y[:, 0:1, :]
        else:
            return None

    def forward(self, x, states=None):
        # 1) First, check for NaN/Inf in the raw input
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"→ [DenseNet.forward] Input x contains NaN/Inf, "
                  f"x.min()={x.min():.3e}, x.max()={x.max():.3e}")

        # 2) Check for extremely large values in x
        max_abs = x.abs().max().item()
        if max_abs > 1e3:
            print(f"→ [DenseNet.forward] Input x is very large: "
                  f"max abs(x) = {max_abs:.3e}")

        # 3) Now clamp x so DenseNet never sees values beyond ±10
        x = x.clamp(min=-10.0, max=10.0)

        # 4) Run through each sub‐layer, printing if any produce NaN/Inf
        out = x
        for idx, layer in enumerate(self.model):
            out = layer(out)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print(f"→ [DenseNet.forward] NaN/Inf after layer {idx} "
                      f"({layer.__class__.__name__})")

        # 5) If you have a final tanh, check it too
        if self.tanh is not None:
            out = self.tanh(out)
            if torch.isnan(out).any() or torch.isinf(out).any():
                print("→ [DenseNet.forward] NaN/Inf after tanh()")

        return out, states



class IdealModel(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.placeholder = nn.Parameter(torch.ones(1, device=device))

    def get_starting_states(self, batch_size, y=None):
        return y

    def forward(self, x, states=None):
        return states * self.placeholder, states


class BlockDenseNet(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.n_networks = kwargs.get('n_networks')
        hidden_size = kwargs.get('hidden_size')
        self.output_size = output_size
        n_layers = kwargs.get('n_layers')

        self.model = nn.ModuleList([DenseNet(input_size, output_size, device, hidden_size=hidden_size, n_layers=n_layers) for _ in range(self.n_networks)])

    def get_starting_states(self, batch_size, x=None):
        return None

    def forward(self, x, states=None):
        # x.shape = [batch_size, seq_len, n_networks, input_size]
        # so our output shape should be [batch_size, seq_len, n_networks*output_size]
        # out = torch.zeros((x.shape[0], x.shape[1], self.n_networks*self.output_size), dtype=torch.float, device=self.device)
        # for i in range(x.shape[1]):
        #     out[:, i:i + 1, :] = torch.cat([subnet(x[:, i:i + 1, sub_idx, :])[0] for sub_idx, subnet in enumerate(self.model)], dim=2)
        # return out, states

        batch_size, sequence_length, num_networks, input_size = x.shape

        # Reshape to (num_networks, batch_size * sequence_length, input_size) for parallel processing
        x = x.permute(2, 0, 1, 3).reshape(num_networks, -1, input_size)

        # Apply each sub-network in parallel using list comprehension and stacking results
        outputs = torch.stack([self.model[i](x[i])[0] for i in range(num_networks)], dim=2)

        # Reshape back to the original batch and sequence dimensions
        out = outputs.reshape(batch_size, sequence_length, num_networks * self.output_size)

        return out, states

class MatMul(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        self.weight = nn.Parameter(torch.randn(input_size, output_size, device=device))

    def get_starting_states(self, batch_size, y=None):
        return None

    def forward(self, x, states=None):
        return torch.matmul(x, self.weight), states

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
        # return self.model(x, states[0])

        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)  # todo size of states to calc input size
        for i in range(x.shape[1]):
            model_input = torch.cat([x[:, i, :], states[1].flatten(start_dim=1)], dim=1).unsqueeze(1)  # todo torch.zeros_like for naive
            out[:, i:i+1, :], states[0] = self.model(model_input, states[0])
        return out, None

    def get_model(self, input_size, output_size, config):
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
        elif config['model_type'] == 'CNN':
            modelclass = CNN
        elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
            modelclass = RNN
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
        # print(f"get_starting_states: y.shape = {y.shape}")
        theta = y[:, 0, :]
        d_theta = (y[:, 1, :] - y[:, 0, :]) * SR
        states = torch.stack([theta, d_theta], dim=2)
        # print(f"get_starting_states: y.shape = {y.shape}")
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
class ActivationGRU(TimeSeriesRegressor):
    def __init__(self, input_size, output_size, device, **kwargs):
        super().__init__(input_size, output_size, device)
        # Grab hyperparameters from kwargs:
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.n_layers    = kwargs.get('n_layers', 1)
        # Build a single‐layer GRU:
        self.gru = nn.GRU(
            input_size,                           # EMG channels (e.g. 8)
            self.hidden_size,                     # size of the GRU hidden state
            self.n_layers,                        # number of stacked GRU layers
            batch_first=True                      # (batch, seq_len, input_size)
        )
        self.layernorm = nn.LayerNorm(self.hidden_size)
        # Readout each time‐step from the last hidden dimension → joint angles:
        self.fc  = nn.Linear(self.hidden_size, output_size)

    def get_starting_states(self, batch_size, y=None):
        # Initialize h₀ as zeros: shape = (n_layers, batch_size, hidden_size)
        return torch.zeros(
            self.n_layers, 
            batch_size, 
            self.hidden_size, 
            dtype=torch.float,
            device=self.device
        )

    def forward(self, x, h0):
        # x:  (batch, seq_len, input_size)
        # h0: (n_layers, batch, hidden_size)
        out, h_n = self.gru(x, h0)      # out: (batch, seq_len, hidden_size)
        out = self.fc(out)              # out: (batch, seq_len, output_size)
        out = nn.functional.dropout(out, p=0.2, training=self.training)
        return out, h_n


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

         # Check if this is an interaction model (has force + material inputs)
        self.is_interaction_model = kwargs.get('is_interaction_model', False)

        self.device = device

        # Input size for activation model
        # For interaction: EMG (8) + Force (5) + Material (1) = 14
        # For free-space: EMG (8) only
        activation_input_size = input_size

        self.activation_model = self.get_model(activation_input_size, output_size * 2, self.activation_model_config)
        self.sigmoid = nn.Sigmoid()
        self.muscle_model = PhysMuscleModel(output_size * 2, output_size * 4, self.device, **self.muscle_model_config) if self.muscle_model_config['model_type'] == 'PhysMuscleModel' else NNMuscleModel(output_size * 2, output_size * 4, self.device, **self.muscle_model_config)
        self.joint_model = PhysJointModel(output_size * 4, output_size, self.device, **self.joint_model_config) if self.joint_model_config['model_type'] == 'PhysJointModel' else NNJointModel(output_size * 4, output_size, self.device, **self.joint_model_config)

        # Store intermediate activations for loss computation
        self.last_muscle_activations = None

    def get_model(self, input_size, output_size, config):
        if config['model_type'] == 'DenseNet':
            modelclass = DenseNet
        elif config['model_type'] == 'CNN':
            modelclass = CNN
        elif config['model_type'] in ['RNN', 'LSTM', 'GRU']:
            modelclass = RNN
        elif config['model_type'] == 'MatMul':
            modelclass = MatMul
        elif config['model_type'] == 'PhysMuscleModel':
            modelclass = PhysMuscleModel
        elif config['model_type'] == 'PhysJointModel':
            modelclass = PhysJointModel
        else:
            raise ValueError(f'Unknown model type {config["model_type"]}')
        return modelclass(input_size, output_size, self.device, **config)

    def get_starting_states(self, batch_size, y=None):
        return [self.activation_model.get_starting_states(batch_size, y),  self.muscle_model.get_starting_states(batch_size, y), self.joint_model.get_starting_states(batch_size, y)]

    def forward(self, x, states):
        out = torch.zeros((x.shape[0], x.shape[1], self.output_size), dtype=torch.float, device=self.device)
        muscle_activations_sequence = []

        for i in range(x.shape[1]):
            activation_out, states[0] = self.activation_model(x[:, i:i+1, :], states[0])
            activation_out = self.sigmoid(activation_out)

            # Store muscle activations for loss computation
            muscle_activations_sequence.append(activation_out)

            muscle_out, states[1] = self.muscle_model(activation_out, [states[1], states[2][0]])
            out[:, i:i+1, :], states[2] = self.joint_model(muscle_out, states[2])

        # Store the full sequence of muscle activations
        self.last_muscle_activations = torch.cat(muscle_activations_sequence, dim=1)

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
        elif config['model_type'] == 'BlockDenseNet':
            modelclass = BlockDenseNet
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


class TimeSeriesRegressorWrapper:
    def __init__(self, input_size, output_size, device, n_epochs, learning_rate, weight_decay, warmup_steps, model_type, **kwargs):

        kwargs.update({'model_type': model_type})

        if model_type == 'DenseNet':
            self.model = DenseNet(input_size, output_size, device, **kwargs)
        elif model_type == 'CNN':
            self.model = CNN(input_size, output_size, device, **kwargs)
        elif model_type == 'GRU':
            self.model = ActivationGRU(input_size, output_size, device, **kwargs)
        # elif model_type in ['RNN', 'LSTM', 'GRU']:
        #     self.model = RNN(input_size, output_size, device, **kwargs)
        elif model_type == 'ModularModel':
            self.model = ModularModel(input_size, output_size, device, **kwargs)
        elif model_type == 'IdealModel':
            self.model = IdealModel(input_size, output_size, device, **kwargs)
        elif model_type == 'CompensationModel':
            self.model = CompensationModel(input_size, output_size, device, **kwargs)
        else:
            raise ValueError(f'Unknown model type {model_type}')

        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, amsgrad=True)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, threshold_mode='rel', threshold=0.01)


        self.n_epochs = n_epochs
        self.warmup_steps = warmup_steps # todo
        # self.warmup_steps = kwargs.get('seq_len') - 1


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))  # note that we load to cpu
        return self

    def train(self):
        self.model.train()
        return

    def eval(self):
        self.model.eval()
        return
    
    def weighted_mse_loss(self, pred, target, weights):
        # pred/target: (batch, seq_len, output_size)
        # weights: (output_size,) or broadcastable to pred/target shape
        loss = (pred - target) ** 2
        weighted_loss = loss * weights  # Will broadcast weights to last dim
        return weighted_loss.mean()

    def train(self):
        self.model.train()
        return

    def eval(self):
        self.model.eval()
        return

    # def train_one_epoch(self, dataloader):
    #     self.model.train()
    #     epoch_loss = 0
    #     for x, y in dataloader:
    #         states = self.model.get_starting_states(dataloader.batch_size, y)
    #         outputs, states = self.model(x, states)
# 
    #         loss = self.criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])
# 
    #         self.optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4)
    #         self.optimizer.step()
    #         epoch_loss += loss
# 
    #         if torch.any(torch.isnan(loss)):
    #             print('NAN Loss!')
# 
    #     return epoch_loss.item() / len(dataloader)

    def train_one_epoch(self, dataloader):
        self.model.train()
        epoch_loss = 0.0
        self.dof_min = torch.tensor([0, 0, 0, 0, 0, -120], dtype=torch.float32, device=self.model.device)  
        self.dof_max = torch.tensor([120, 120, 120, 120, 120, 0], dtype=torch.float32, device=self.model.device)

        # 1) Create one global counter before either loop
        global_batch_idx = 0

        output_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.model.device)

        # 2) First pass: just check x/y for NaNs or Infs
        for x, y in dataloader:
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"[GlobalBatch {global_batch_idx}] NaN/Inf in x!  "
                    f"x.min()={x.min():.3e}, x.max()={x.max():.3e}")
                # we skip reporting any training on this batch
            elif torch.isnan(y).any() or torch.isinf(y).any():
                print(f"[GlobalBatch {global_batch_idx}] NaN/Inf in y!  "
                    f"y.min()={y.min():.3e}, y.max()={y.max():.3e}")
            # advance the global counter, even if we “skipped” training
            global_batch_idx += 1

        # 3) Second pass: actually do the forward/backward on the same data,
        #    but continue incrementing the same global counter
        for x, y in dataloader:
            states = self.model.get_starting_states(dataloader.batch_size, y)
            outputs, states = self.model(x, states)
            y_targets = y[:, self.warmup_steps:]

            # Prepare thresholds for "extremes" per DoF
            low_thresh = self.dof_min + 0.3 * (self.dof_max - self.dof_min)
            high_thresh = self.dof_max - 0.3 * (self.dof_max - self.dof_min)
            # Reshape for broadcasting
            low_thresh = low_thresh.view(1, 1, -1)
            high_thresh = high_thresh.view(1, 1, -1)

            # Compute weights for each element in the batch/seq/DoF
            weights = torch.where((y_targets < low_thresh) | (y_targets > high_thresh), 2.0, 1.0)
            weights = weights * output_weights.view(1, 1, -1)

            # Weighted loss
            loss = ((outputs[:, self.warmup_steps:] - y_targets) ** 2 * weights).mean()


            # Penalize Mean
            per_dof_means = y_targets.mean(dim=(0,1), keepdim=True).detach()  # [1, 1, 6]
            mean_penalty = torch.abs(outputs[:, self.warmup_steps:] - per_dof_means).mean()
            alpha = 0.01
            loss = loss + alpha * mean_penalty

            # Per-DoF Variance Matching
            target_var_per_dof = y_targets.var(dim=1)  # Shape: [batch_size, 6 DoFs]
            pred_var_per_dof = outputs[:, self.warmup_steps:].var(dim=1)  # Shape: [batch_size, 6 DoFs]

            var_matching_loss = ((pred_var_per_dof - target_var_per_dof) ** 2).mean()
            gamma = 0.01  # Weight for variance matching term
            loss = loss + gamma * var_matching_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            epoch_loss += loss.detach().item()

            if torch.any(torch.isnan(loss)):
                print(f"[GlobalBatch {global_batch_idx}] NAN Loss!")

            global_batch_idx += 1
        return epoch_loss / len(dataloader)


    def train_one_epoch_enhanced(self, dataloader, use_enhanced_loss=False):
        """
        Enhanced training loop with optional physiology loss
        """
        self.model.train()
        epoch_loss = 0.0
        position_loss_sum = 0.0
        physiology_loss_sum = 0.0
        
        # DOF limits for joint angle constraints (from your existing code)
        self.dof_min = torch.tensor([0, 0, 0, 0, 0, -120], dtype=torch.float32, device=self.model.device)  
        self.dof_max = torch.tensor([120, 120, 120, 120, 120, 0], dtype=torch.float32, device=self.model.device)
        
        output_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=self.model.device)

        for batch_data in dataloader:
            # Handle different batch formats
            if len(batch_data) >= 3:
                x, y, force_data = batch_data[:3]
            else:
                x, y = batch_data[:2]
                force_data = None
            
            states = self.model.get_starting_states(dataloader.batch_size, y)
            outputs, final_states = self.model(x, states)
            
            # Get muscle activations if available (for ModularModel)
            muscle_activations = None
            if hasattr(self.model, 'last_muscle_activations'):
                muscle_activations = self.model.last_muscle_activations
            
            # Use enhanced loss if requested and data is available
            if use_enhanced_loss and muscle_activations is not None and force_data is not None:

                # print("Using simplified enhanced loss...")
                assert muscle_activations is not None, "Muscle_activations is None!"
                assert force_data is not None, "Force_data is None!"

                loss, pos_loss, phys_loss = simplified_enhanced_loss(
                    outputs, y, muscle_activations, force_data, self.warmup_steps
                )

                position_loss_sum += pos_loss
                physiology_loss_sum += phys_loss
                
            else:
                # Standard loss computation (from your existing train_one_epoch)
                y_targets = y[:, self.warmup_steps:]
                
                # Your existing loss calculation logic
                loss = ((outputs[:, self.warmup_steps:] - y_targets) ** 2).mean()

                # Your existing regularization terms
                y_mean = y_targets.mean().detach()
                mean_penalty = torch.abs(outputs[:, self.warmup_steps:] - y_mean).mean()
                pred_var = outputs[:, self.warmup_steps:].var()
                
                alpha, beta = 0.01, 0.01
                loss = loss + alpha * mean_penalty - beta * pred_var

            # Backward pass (same for both)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.detach().item()

            if torch.any(torch.isnan(loss)):
                print(f"NAN Loss detected!")

        avg_epoch_loss = epoch_loss / len(dataloader)
        
        # Return detailed loss info if using enhanced loss
        if use_enhanced_loss:
            return {
                'total_loss': avg_epoch_loss,
                'position_loss': position_loss_sum / len(dataloader),
                'physiology_loss': physiology_loss_sum / len(dataloader)
            }
        else:
            return avg_epoch_loss
    

    def predict(self, test_set, features, targets):
        self.model.eval()
        x = torch.tensor(test_set.loc[:, features].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        y = torch.tensor(test_set.loc[:, targets].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        # print(f"predict: input y shape = {y.shape}")
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
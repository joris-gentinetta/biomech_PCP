import torch
import torch.nn as nn
import torch.optim as optim


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, model_type='LSTM'):
        super(RNNClassifier, self).__init__()
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


class CNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CNNClassifier, self).__init__()
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


class TorchTimeSeriesClassifier:
    def __init__(self, input_size, hidden_size, output_size, n_epochs, seq_len, learning_rate, warmup_steps, num_layers, model_type):
        self.model_type = model_type
        self.device = torch.device("cpu")
        if self.model_type == 'CNN':
            self.model = CNNClassifier(input_size, hidden_size, output_size, num_layers)
        else:
            self.model = RNNClassifier(input_size, hidden_size, output_size, num_layers, self.model_type)
        self.train_criterion = nn.MSELoss(reduction='mean')
        self.eval_criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # self.sigmoid = nn.Sigmoid()
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

            if self.model_type == 'LSTM':
                states = (torch.zeros(self.model.num_layers, dataloader.batch_size, self.model.hidden_size), torch.zeros(self.model.num_layers, dataloader.batch_size, self.model.hidden_size))
            else:
                states = torch.zeros(self.model.num_layers, dataloader.batch_size, self.model.hidden_size)


            # Forward pass
            outputs, states = self.model(x, states)
            # if self.model_type == 'LSTM':
            #     states = (states[0].detach(), states[1].detach()) # Detach states to prevent backprop through time over the entire sequence
            # else:
            #     states = states.detach()
            # start = self.warmup_steps if i == 0 or self.model_type == 'CNN' else 0
            loss = self.train_criterion(outputs[:, self.warmup_steps:], y[:, self.warmup_steps:])

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def predict(self, test_set, features):
        self.model.eval()
        X_test = torch.tensor(test_set.loc[:, features].values, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_type == 'LSTM':
                states = (torch.zeros(self.model.num_layers, 1, self.model.hidden_size), torch.zeros(self.model.num_layers, 1, self.model.hidden_size))
            else:
                states = torch.zeros(self.model.num_layers, 1, self.model.hidden_size)

            y_pred, _ = self.model(X_test, states=states)
            # y_pred = self.sigmoid(y_pred).squeeze()
        return y_pred

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self



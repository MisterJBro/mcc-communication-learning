import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super(Policy, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(in_dim, hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x, lens = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        out, (_, _) = self.rnn(x)
        out = out.squeeze()
        probs = F.softmax(self.fc(F.leaky_relu(out)), dim=-1)
        return Categorical(probs=probs)

    def forward_with_state(self, x, state):
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        probs = F.softmax(self.fc(F.leaky_relu(x)), dim=-1)
        return Categorical(probs=probs), (h_n, c_n)

    def packed_forward(self, x):
        out, _ = self.rnn(x)
        out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.cat([out[i, :x].reshape(-1, self.hidden_dim)
                         for i, x in enumerate(lens)])
        probs = self.fc(F.leaky_relu(out))
        return Categorical(probs=probs)


class ValueFunction(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(ValueFunction, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(in_dim, hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, lens = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        out, (_, _) = self.rnn(x)
        out = out.squeeze()
        out = self.fc(F.leaky_relu(out)).squeeze()
        return out

    def full_forward(self, x):
        out, _ = self.rnn(x)
        out = out.squeeze()
        out = self.fc(F.leaky_relu(out)).reshape(-1)
        return out

    def packed_forward(self, x):
        out, _ = self.rnn(x)
        out, lens = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.cat([out[i, :x].reshape(-1, self.hidden_dim)
                         for i, x in enumerate(lens)])
        out = self.fc(F.leaky_relu(out)).reshape(-1)
        return out


class ValueFunctionNonRecurrent(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ValueFunction, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.model(x)

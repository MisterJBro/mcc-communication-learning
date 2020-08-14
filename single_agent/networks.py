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

        self.fc1 = nn.Sequential(
            nn.Linear(100, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
        )
        self.rnn = nn.LSTM(64, hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.reshape(batch_size, seq_len, -1)
        x = self.fc1(x)
        out, (_, _) = self.rnn(x)
        out = out.reshape(-1, self.hidden_dim)
        probs = self.fc2(out)
        return Categorical(probs=probs)

    def with_state(self, x, state):
        x = x.reshape(1, -1)
        x = self.fc1(x).unsqueeze(0)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        probs = self.fc2(x)
        return Categorical(probs=probs), (h_n, c_n)


class ValueFunction(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers):
        super(ValueFunction, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.LeakyReLU()
        )
        self.rnn = nn.LSTM(32, hidden_dim,
                           num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.reshape(batch_size, seq_len, -1)
        x = self.fc1(x)
        out, (_, _) = self.rnn(x)
        out = out.reshape(-1, self.hidden_dim)
        out = self.fc2(out).reshape(-1)
        return out

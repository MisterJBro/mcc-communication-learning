import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, in_dim, out_dim, fc_hidden=64, rnn_hidden=128, num_layers=1):
        super(Policy, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, fc_hidden),
            nn.ELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ELU(),
        )
        self.rnn = nn.LSTM(fc_hidden, rnn_hidden,
                           num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_hidden, out_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.reshape(batch_size, seq_len, -1)
        x = self.fc1(x)
        out, (_, _) = self.rnn(x)
        out = out.reshape(-1, out.size(-1))
        probs = self.fc2(out)
        return Categorical(probs=probs)

    def with_state(self, x, state):
        x = self.fc1(x)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        probs = self.fc2(x)
        return Categorical(probs=probs), (h_n, c_n)


class ValueFunction(nn.Module):
    def __init__(self, in_dim, fc_hidden=64, rnn_hidden=128, num_layers=1):
        super(ValueFunction, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, fc_hidden),
            nn.LeakyReLU()
        )
        self.rnn = nn.LSTM(fc_hidden, rnn_hidden,
                           num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_hidden, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        x = x.reshape(batch_size, seq_len, -1)
        x = self.fc1(x)
        out, (_, _) = self.rnn(x)
        out = out.reshape(-1, out.size(-1))
        out = self.fc2(out).reshape(-1)
        return out

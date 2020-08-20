import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Network(nn.Module):
    def __init__(self, in_dim, action_num, message_num, fc_hidden=64, rnn_hidden=128, num_layers=1):
        super(Network, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, fc_hidden),
            nn.ELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ELU(),
        )
        self.rnn = nn.LSTM(fc_hidden, rnn_hidden,
                           num_layers=num_layers, batch_first=True)

        self.action = nn.Sequential(
            nn.Linear(rnn_hidden, action_num),
            nn.Softmax(dim=-1)
        )
        self.message = nn.Sequential(
            nn.Linear(rnn_hidden, message_num),
        )
        self.value = nn.Sequential(
            nn.Linear(rnn_hidden, 1),
        )

    def next_action(self, x, state):
        x = self.mlp(x)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]

        return action_dist, message, (h_n, c_n)

    def forward(self, x):
        x = self.mlp(x)
        x, _ = self.rnn(x)
        x = x.reshape(-1, x.size(-1))

        action_dists = Categorical(probs=self.action(x))
        messages = F.gumbel_softmax(self.message(x), tau, hard=True)
        values = self.value(x)
        return action_dists, messages, values

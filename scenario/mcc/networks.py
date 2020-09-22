import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, in_dim, action_num, symbol_num, fc_hidden=64, rnn_hidden=128, num_layers=1, tau=1.0):
        super(Policy, self).__init__()
        self.tau = tau

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, fc_hidden),
            nn.ELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ELU(),
        )

        self.rnn = nn.LSTM(fc_hidden+symbol_num, rnn_hidden,
                           num_layers=num_layers, batch_first=True)

        self.action = nn.Sequential(
            nn.Linear(rnn_hidden, action_num),
            nn.Softmax(dim=-1)
        )
        self.message = nn.Sequential(
            nn.Linear(rnn_hidden, symbol_num),
        )
        self.value = nn.Sequential(
            nn.Linear(rnn_hidden, 1),
        )

    def next_action(self, x, c, state):
        x = self.mlp(x)
        x = torch.cat((x, c), 2)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        action_dist = Categorical(probs=self.action(x))
        message = F.gumbel_softmax(self.message(x), self.tau, hard=True)

        return action_dist, message, (h_n, c_n)

    def forward(self, x, c):
        x = self.tail(x, c)

        action_dists = Categorical(probs=self.action(x))
        messages = F.gumbel_softmax(self.message(x), self.tau, hard=True)
        values = self.value(x)
        return action_dists, messages, values

    def value_only(self, x, c):
        x = self.tail(x, c)
        return self.value(x)

    def action_only(self, x, c):
        x = self.tail(x, c)
        return Categorical(probs=self.action(x))

    def message_only(self, x, c):
        x = self.tail(x, c)
        return F.gumbel_softmax(self.message(x), self.tau, hard=True)

    def tail(self, x, c):
        x = self.mlp(x)
        x = torch.cat((x, c), 2)
        x, _ = self.rnn(x)
        x = x.reshape(-1, x.size(-1))

        return x


class Speaker(nn.Module):
    def __init__(self, in_dim, action_num, symbol_num, fc_hidden=64, rnn_hidden=128, num_layers=1, tau=1.0):
        super(Speaker, self).__init__()
        self.tau = tau

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
            nn.Linear(rnn_hidden, symbol_num),
        )
        self.value = nn.Sequential(
            nn.Linear(rnn_hidden, 1),
        )

    def set_requires_grad(self, grad):
        for params in self.mlp.parameters():
            params.requires_grad = grad
        for params in self.rnn.parameters():
            params.requires_grad = grad

    def next_action(self, x, state):
        x = self.mlp(x)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        action_dist = Categorical(probs=self.action(x))
        message = F.gumbel_softmax(self.message(x), self.tau, hard=True)

        return action_dist, message, (h_n, c_n)

    def forward(self, x):
        x = self.tail(x)

        action_dists = Categorical(probs=self.action(x))
        messages = F.gumbel_softmax(self.message(x), self.tau, hard=True)
        values = self.value(x)
        return action_dists, messages, values

    def value_only(self, x):
        x = self.tail(x)
        return self.value(x)

    def action_only(self, x):
        x = self.tail(x)
        return Categorical(probs=self.action(x))

    def message_only(self, x):
        x = self.tail(x)
        return F.gumbel_softmax(self.message(x), self.tau, hard=True)

    def tail(self, x):
        x = self.mlp(x)
        x, _ = self.rnn(x)
        x = x.reshape(-1, x.size(-1))

        return x


class Listener(nn.Module):
    def __init__(self, in_dim, action_num, symbol_num, fc_hidden=64, rnn_hidden=128, num_layers=1, tau=1.0):
        super(Listener, self).__init__()
        self.tau = tau

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, fc_hidden),
            nn.ELU(),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ELU(),
        )

        self.rnn = nn.LSTM(fc_hidden+symbol_num, rnn_hidden,
                           num_layers=num_layers, batch_first=True)

        self.action = nn.Sequential(
            nn.Linear(rnn_hidden, action_num),
            nn.Softmax(dim=-1)
        )
        self.value = nn.Sequential(
            nn.Linear(rnn_hidden, 1),
        )

    def next_action(self, x, c, state):
        x = self.mlp(x)
        x = torch.cat((x, c), 2)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        action_dist = Categorical(probs=self.action(x))

        return action_dist, (h_n, c_n)

    def forward(self, x, c):
        x = self.tail(x, c)

        action_dists = Categorical(probs=self.action(x))
        values = self.value(x)
        return action_dists, values

    def value_only(self, x, c):
        x = self.tail(x, c)
        return self.value(x)

    def action_only(self, x, c):
        x = self.tail(x, c)
        return Categorical(probs=self.action(x))

    def tail(self, x, c):
        x = self.mlp(x)
        x = torch.cat((x, c), 2)
        x, _ = self.rnn(x)
        x = x.reshape(-1, x.size(-1))

        return x


class Normal(nn.Module):
    def __init__(self, in_dim, action_num, symbol_num, fc_hidden=64, rnn_hidden=128, num_layers=1):
        super(Normal, self).__init__()

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
        self.value = nn.Sequential(
            nn.Linear(rnn_hidden, 1),
        )

    def next_action(self, x, state):
        x = self.mlp(x)
        _, (h_n, c_n) = self.rnn(x, state)
        x = h_n[-1]
        action_dist = Categorical(probs=self.action(x))

        return action_dist, (h_n, c_n)

    def forward(self, x):
        x = self.tail(x)

        action_dists = Categorical(probs=self.action(x))
        values = self.value(x)
        return action_dists, values

    def value_only(self, x):
        x = self.tail(x)
        return self.value(x)

    def action_only(self, x):
        x = self.tail(x)
        return Categorical(probs=self.action(x))

    def tail(self, x):
        x = self.mlp(x)
        x, _ = self.rnn(x)
        x = x.reshape(-1, x.size(-1))

        return x


class ActionValue(nn.Module):
    def __init__(self, in_dim, agents, action_dim, batch_size, steps, fc_hidden=128):
        super(ActionValue, self).__init__()
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.steps = steps

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
        )

        self.rnn = nn.LSTM(32+agents, 64,
                           num_layers=1, batch_first=True)

        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x, a):
        x = self.mlp(x)
        x = torch.cat([x, a.float()], dim=-1)
        x, _ = self.rnn(x)
        x = x.reshape(-1, x.size(-1))
        x = self.out(x)
        return x

    def all_actions_c(self, x, act_e):
        """ Calculates all q values for all action of red collector."""
        all = []
        for a in range(self.action_dim):
            act_c = torch.ones(
                act_e.size(), dtype=torch.int32, device=act_e.device)*a
            acts = torch.stack([act_c, act_e], dim=1).reshape(
                self.batch_size, self.steps, -1)

            y = self.mlp(x)
            y = torch.cat([y, acts.float()], dim=-1)
            y, _ = self.rnn(y)
            y = y.reshape(-1, y.size(-1))
            y = self.out(y)
            all.append(y)

        return torch.cat(all, dim=-1)

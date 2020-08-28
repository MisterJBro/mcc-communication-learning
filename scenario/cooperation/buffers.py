import numpy as np
import torch
from scenario.cooperation.buffer import Buffer


class Buffers:
    def __init__(self, batch_size, size, obs_dim, gamma, lam, symbol_num):
        self.batch_size = batch_size

        self.buffer_c = Buffer(batch_size, size,
                               obs_dim, gamma, lam, symbol_num)
        self.buffer_g = Buffer(batch_size, size,
                               obs_dim, gamma, lam, symbol_num)

    def clear(self):
        self.buffer_c.clear()
        self.buffer_g.clear()

    def store(self, obs, acts, rews, msg):
        self.buffer_c.store(obs[0], acts[:, 0], rews, msg=msg)
        self.buffer_g.store(obs[1], acts[:, 1], rews)

    def expected_returns(self):
        self.buffer_c.expected_returns()
        self.buffer_g.expected_returns()

    def advantage_estimation(self, vals):
        self.buffer_c.advantage_estimation(
            vals[0], np.zeros((self.batch_size, 1)))
        self.buffer_g.advantage_estimation(
            vals[1], np.zeros((self.batch_size, 1)))

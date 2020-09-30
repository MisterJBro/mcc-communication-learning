import numpy as np
import torch
from scenario.competition.buffer import Buffer


class Buffers:
    def __init__(self, batch_size, size, obs_dim, gamma, lam, symbol_num):
        self.batch_size = batch_size
        self.size = size

        self.buffer_c = Buffer(batch_size, size,
                               obs_dim, gamma, lam, symbol_num)
        self.buffer_e = Buffer(batch_size, size,
                               obs_dim, gamma, lam, symbol_num)
        self.backprop_msg = None

    def clear(self):
        self.buffer_c.clear()
        self.buffer_e.clear()
        self.backprop_msg = None

    def store(self, obs, acts, rews_c, rews_e):
        self.buffer_c.store(obs[0], acts[:, 0], rews_c)
        self.buffer_e.store(obs[1], acts[:, 1], rews_e)

    def expected_returns(self):
        self.buffer_c.expected_returns()
        self.buffer_e.expected_returns()

    def advantage_estimation(self, vals):
        self.buffer_c.advantage_estimation(
            vals[0], np.zeros((self.batch_size, 1)))
        self.buffer_e.advantage_estimation(
            vals[1], np.zeros((self.batch_size, 1)))

    def standardize_adv(self):
        self.buffer_c.standardize_adv()
        self.buffer_e.standardize_adv()

    def get_tensors(self, device):
        obs_c = torch.as_tensor(
            self.buffer_c.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_c = torch.as_tensor(
            self.buffer_c.act_buf, dtype=torch.int32, device=device).reshape(-1)
        ret_c = torch.as_tensor(
            self.buffer_c.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_c = torch.as_tensor(
            self.buffer_c.adv_buf, dtype=torch.float32, device=device).reshape(-1)

        obs_e = torch.as_tensor(
            self.buffer_e.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_e = torch.as_tensor(
            self.buffer_e.act_buf, dtype=torch.int32, device=device).reshape(-1)
        ret_e = torch.as_tensor(
            self.buffer_e.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_e = torch.as_tensor(
            self.buffer_e.adv_buf, dtype=torch.float32, device=device).reshape(-1)

        return obs_c, act_c, ret_c, adv_c, obs_e, act_e, ret_e, adv_e

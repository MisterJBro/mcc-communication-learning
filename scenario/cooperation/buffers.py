import numpy as np
import torch
from scenario.cooperation.buffer import Buffer


class Buffers:
    def __init__(self, batch_size, size, obs_dim, gamma, lam, symbol_num):
        self.batch_size = batch_size
        self.size = size

        self.buffer_c = Buffer(batch_size, size,
                               obs_dim, gamma, lam, symbol_num)
        self.buffer_g = Buffer(batch_size, size,
                               obs_dim, gamma, lam, symbol_num)
        self.backprop_msg = None
        self.trs_buf = np.empty(
            (self.batch_size, self.size))

    def clear(self):
        self.buffer_c.clear()
        self.buffer_g.clear()
        self.backprop_msg = None
        self.trs_buf = np.empty(
            (self.batch_size, self.size))

    def store(self, obs, acts, rews_c, rews_g, msg, trs):
        if self.backprop_msg is None:
            self.backprop_msg = msg.reshape(self.batch_size, 1, -1)
        else:
            self.backprop_msg = torch.cat(
                (self.backprop_msg, msg.reshape(self.batch_size, 1, -1)), 1)
        self.trs_buf[:, self.buffer_c.ptr] = trs

        self.buffer_c.store(obs[0], acts[:, 0], rews_c)
        self.buffer_g.store(obs[1], acts[:, 1], rews_g)

    def expected_returns(self):
        self.buffer_c.expected_returns()
        self.buffer_g.expected_returns()

    def advantage_estimation(self, vals):
        self.buffer_c.advantage_estimation(
            vals[0], np.zeros((self.batch_size, 1)))
        self.buffer_g.advantage_estimation(
            vals[1], np.zeros((self.batch_size, 1)))

    def standardize_adv(self):
        self.buffer_c.standardize_adv()
        self.buffer_g.standardize_adv()

    def get_tensors(self, device):
        obs_c = torch.as_tensor(
            self.buffer_c.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_c = torch.as_tensor(
            self.buffer_c.act_buf, dtype=torch.int32, device=device).reshape(-1)
        ret_c = torch.as_tensor(
            self.buffer_c.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_c = torch.as_tensor(
            self.buffer_c.adv_buf, dtype=torch.float32, device=device).reshape(-1)

        obs_g = torch.as_tensor(
            self.buffer_g.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_g = torch.as_tensor(
            self.buffer_g.act_buf, dtype=torch.int32, device=device).reshape(-1)
        ret_g = torch.as_tensor(
            self.buffer_g.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_g = torch.as_tensor(
            self.buffer_g.adv_buf, dtype=torch.float32, device=device).reshape(-1)

        msg = self.backprop_msg
        trs = torch.as_tensor(
            self.trs_buf, dtype=torch.long, device=device).reshape(-1)

        return obs_c, act_c, ret_c, adv_c, obs_g, act_g, ret_g, adv_g, msg, trs

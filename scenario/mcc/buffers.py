import numpy as np
import torch
from scenario.mcc.buffer import Buffer


class Buffers:
    def __init__(self, batch_size, size, obs_dim, act_num, gamma, lam, symbol_num, state_dim):
        self.batch_size = batch_size
        self.act_num = act_num
        self.size = size

        self.buffer_c = Buffer(batch_size, size,
                               obs_dim, act_num, gamma, lam, symbol_num)
        self.buffer_g = Buffer(batch_size, size,
                               obs_dim, act_num, gamma, lam, symbol_num)
        self.buffer_e = Buffer(batch_size, size,
                               obs_dim, act_num, gamma, lam, symbol_num)
        self.backprop_msg = None
        self.states = np.empty((batch_size, size) +
                               state_dim, dtype=np.float32)
        self.t = None

    def clear(self):
        self.buffer_c.clear()
        self.buffer_g.clear()
        self.buffer_e.clear()
        self.backprop_msg = None

    def store(self, obs, acts, dsts, rews_c, rews_g, rews_e, msg, states):
        if self.backprop_msg is None:
            self.backprop_msg = msg.reshape(self.batch_size, 1, -1)
        else:
            self.backprop_msg = torch.cat(
                (self.backprop_msg, msg.reshape(self.batch_size, 1, -1)), 1)

        self.states[:, self.buffer_c.ptr] = states

        self.buffer_c.store(obs[0], acts[:, 0], dsts[0], rews_c)
        self.buffer_g.store(obs[1], acts[:, 1], dsts[1], rews_g)
        self.buffer_e.store(obs[2], acts[:, 2], dsts[2], rews_e)

    def expected_returns(self):
        self.buffer_c.expected_returns()
        self.buffer_g.expected_returns()
        self.buffer_e.expected_returns()

    def advantage_estimation(self, vals):
        self.buffer_c.advantage_estimation(
            vals[0], np.zeros((self.batch_size, 1)))
        self.buffer_g.advantage_estimation(
            vals[1], np.zeros((self.batch_size, 1)))
        self.buffer_e.advantage_estimation(
            vals[2], np.zeros((self.batch_size, 1)))

    def standardize_adv(self):
        self.buffer_c.standardize_adv()
        self.buffer_g.standardize_adv()
        self.buffer_e.standardize_adv()

    def get_central_critic_tensors(self, device):
        obs_c = torch.as_tensor(
            self.buffer_c.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_c = torch.as_tensor(
            self.buffer_c.act_buf, dtype=torch.int32, device=device).reshape(-1)
        dst_c = torch.as_tensor(
            self.buffer_c.dst_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, self.act_num)

        obs_g = torch.as_tensor(
            self.buffer_g.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_g = torch.as_tensor(
            self.buffer_g.act_buf, dtype=torch.int32, device=device).reshape(-1)
        dst_g = torch.as_tensor(
            self.buffer_g.dst_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, self.act_num)

        obs_e = torch.as_tensor(
            self.buffer_e.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_e = torch.as_tensor(
            self.buffer_e.act_buf, dtype=torch.int32, device=device).reshape(-1)
        dst_e = torch.as_tensor(
            self.buffer_e.dst_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, self.act_num)

        msg = self.backprop_msg
        states = torch.as_tensor(
            self.states, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)

        return obs_c, act_c, dst_c, obs_g, act_g, dst_g, obs_e, act_e, dst_e, msg, states

    def get_tensors(self, device):
        obs_c = torch.as_tensor(
            self.buffer_c.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_c = torch.as_tensor(
            self.buffer_c.act_buf, dtype=torch.int32, device=device).reshape(-1)
        rew_c = torch.as_tensor(
            self.buffer_c.rew_buf, dtype=torch.float32, device=device).reshape(-1)
        ret_c = torch.as_tensor(
            self.buffer_c.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_c = torch.as_tensor(
            self.buffer_c.adv_buf, dtype=torch.float32, device=device).reshape(-1)
        dst_c = torch.as_tensor(
            self.buffer_c.dst_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, self.act_num)

        obs_g = torch.as_tensor(
            self.buffer_g.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_g = torch.as_tensor(
            self.buffer_g.act_buf, dtype=torch.int32, device=device).reshape(-1)
        ret_g = torch.as_tensor(
            self.buffer_g.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_g = torch.as_tensor(
            self.buffer_g.adv_buf, dtype=torch.float32, device=device).reshape(-1)
        dst_g = torch.as_tensor(
            self.buffer_g.dst_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, self.act_num)

        obs_e = torch.as_tensor(
            self.buffer_e.obs_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)
        act_e = torch.as_tensor(
            self.buffer_e.act_buf, dtype=torch.int32, device=device).reshape(-1)
        rew_e = torch.as_tensor(
            self.buffer_e.rew_buf, dtype=torch.float32, device=device).reshape(-1)
        ret_e = torch.as_tensor(
            self.buffer_e.ret_buf, dtype=torch.float32, device=device).reshape(-1)
        adv_e = torch.as_tensor(
            self.buffer_e.adv_buf, dtype=torch.float32, device=device).reshape(-1)
        dst_e = torch.as_tensor(
            self.buffer_e.dst_buf, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, self.act_num)

        msg = self.backprop_msg
        states = torch.as_tensor(
            self.states, dtype=torch.float32, device=device).reshape(self.batch_size, self.size, -1)

        return obs_c, act_c, rew_c, ret_c, adv_c, dst_c, obs_g, act_g, ret_g, adv_g, dst_g, obs_e, act_e, rew_e, ret_e, adv_e, dst_e, msg, states

import numpy as np
import torch


class Buffer:
    def __init__(self, batch_size, size, obs_dim, gamma, lam, symbol_num):
        self.obs_buf = np.empty((batch_size, size) + obs_dim, dtype=np.float32)
        self.act_buf = np.empty((batch_size, size), dtype=np.float32)
        self.rew_buf = np.empty((batch_size, size), dtype=np.float32)
        self.ret_buf = np.zeros((batch_size, size), dtype=np.float32)
        self.adv_buf = np.zeros((batch_size, size), dtype=np.float32)
        self.msg_buf = np.empty((batch_size, size, symbol_num), dtype=np.float32)

        self.ptr = 0
        self.batch_size = batch_size
        self.size = size
        self.gamma = gamma
        self.lam = lam

    def clear(self):
        self.ptr = 0
        self.ret_buf = np.zeros((self.batch_size, self.size), dtype=np.float32)
        self.adv_buf = np.zeros((self.batch_size, self.size), dtype=np.float32)

    def store(self, obs, act, rew, msg):
        assert self.ptr < self.size, 'Buffer full!'

        self.obs_buf[:, self.ptr] = obs
        self.act_buf[:, self.ptr] = act
        self.rew_buf[:, self.ptr] = rew
        self.msg_buf[:, self.ptr] = msg
        self.ptr += 1

    def expected_returns(self):
        for t in range(self.ptr):
            self.ret_buf[:, 0:self.ptr-t] += self.gamma**(t+1) * \
                self.rew_buf[:, t:self.ptr]

    def advantage_estimation(self, values, last_val):
        rews = np.concatenate([self.rew_buf, last_val], axis=1)
        vals = np.concatenate([values, last_val], axis=1)

        deltas = rews[:, :-1] + self.gamma*vals[:, 1:] - vals[:, :-1]

        for t in range(self.ptr):
            b = (self.lam*self.gamma)**(t+1) * deltas[:, t:self.ptr]
            self.adv_buf[:, :self.ptr-t] += b

    def standardize_adv(self):
        adv = self.adv_buf[:self.ptr]
        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        self.adv_buf[:self.ptr] = (adv-adv_mean)/adv_std

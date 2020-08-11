import numpy as np

class Buffer:
    def __init__(self, size, obs_dim, gamma, lam):
        self.obs_buf = np.empty((size,) + obs_dim, dtype=np.float32)
        self.act_buf = np.empty(size, dtype=np.float32)
        self.rew_buf = np.empty(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.last_ptr = 0
        self.size = size
        self.gamma = gamma
        self.lam = lam

    def clear(self):
        self.ptr = 0
        self.last_ptr = 0
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)

    def store(self, obs, act, rew):
        assert self.ptr < self.size, 'Buffer full!'

        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.ptr += 1

    def expected_returns(self):
        a = 0
        for t in range(self.last_ptr, self.ptr):
            a += 1
            self.ret_buf[self.last_ptr:self.ptr-a+1] += self.gamma**a * \
                self.rew_buf[t:self.ptr]

    def advantage_estimation(self, values, last_val):
        rews = np.append(self.rew_buf[self.last_ptr:self.ptr], last_val)
        vals = np.append(values, last_val)

        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        a = 0
        for t in range(self.last_ptr, self.ptr):
            a += 1
            b = (self.lam*self.gamma)**a * deltas[a-1:self.ptr-self.last_ptr]
            self.adv_buf[self.last_ptr:self.ptr-a+1] += b

    def standardize_adv(self):
        adv = self.adv_buf[:self.ptr]
        adv_mean = np.mean(adv)
        adv_std = np.std(adv)
        self.adv_buf[:self.ptr] = (adv-adv_mean)/adv_std

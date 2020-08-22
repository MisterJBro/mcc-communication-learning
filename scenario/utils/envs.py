from multiprocessing.dummy import Pool as ThreadPool
import gym
import numpy as np
import time


class Envs():
    def __init__(self, num):
        self.num = num
        self.envs = [gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
                              red_guides=1, blue_collector=0, seed=seed) for seed in range(num)]
        self.pool = ThreadPool(num)

    def reset(self):
        for x in range(self.num):
            self.envs[x].reset()

    def step(self, acts):
        o_s, r_s, d_s, i_s = [], [], [], []
        results = self.pool.starmap(self.env_step, zip(self.envs, acts))
        for o, r, d, i in results:
            o_s.append(o)
            r_s.append(r)
            d_s.append(d)
            i_s.append(i)
        return o_s, r_s, d_s, i_s

    def env_step(self, env, act):
        return env.step(act)

    def close(self):
        for x in range(self.num):
            self.envs[x].close()
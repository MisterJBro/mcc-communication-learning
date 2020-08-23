from multiprocessing.dummy import Pool
import gym
import numpy as np
import time


class Envs():
    def __init__(self, num):
        self.num = num
        self.envs = [gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
                              red_guides=1, blue_collector=0, seed=seed) for seed in range(num)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.agents_num = self.envs[0].agents_num

    def reset(self):
        return [self.envs[x].reset() for x in range(self.num)]

    def step(self, acts):
        o_s, r_s, d_s, i_s = [], [], [], []

        for x in range(self.num):
            o, r, d, i = self.envs[x].step(acts[x])
            o_s.append(o)
            r_s.append(r)
            d_s.append(d)
            i_s.append(i)
        return o_s, r_s, d_s, i_s

    def close(self):
        for x in range(self.num):
            self.envs[x].close()

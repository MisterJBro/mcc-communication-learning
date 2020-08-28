from multiprocessing.dummy import Pool
import gym
import numpy as np
import time


class Envs():
    def __init__(self, num, red_guides=1, red_collector=1, blue_guides=0, blue_collector=1, view_radius=2, competition=False,
                 game_length=500):
        assert num >= 1, 'Minimum 1 Env in Envs class, set num >= 1!'

        self.num = num
        self.envs = [gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
                              red_guides=red_guides, red_collector=red_collector, blue_guides=blue_guides, blue_collector=blue_collector, view_radius=view_radius, competition=competition,
                              game_length=game_length, seed=seed) for seed in range(num)]
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
        return np.array(o_s), np.array(r_s), np.array(d_s), np.array(i_s)

    def close(self):
        for x in range(self.num):
            self.envs[x].close()

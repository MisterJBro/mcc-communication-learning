import numpy as np
import torch
import pickle
import copy


class PolicyData:
    def __init__(self, time, rating):
        self.time = time
        self.rating = rating


class RedPolicy(PolicyData):
    def __init__(self, c_dict, g_dict, time, rating):
        super().__init__(time, rating)

        self.c_dict = copy.deepcopy(c_dict)
        self.g_dict = copy.deepcopy(g_dict)


class BluePolicy(PolicyData):
    def __init__(self, e_dict, time, rating):
        super().__init__(time, rating)

        self.e_dict = copy.deepcopy(e_dict)


class Menagerie:
    def __init__(self, collector, guide, enemy, policy_update_interval=20):
        self.counter = 0
        self.collector = collector
        self.guide = guide
        self.enemy = enemy
        self.policy_update_interval = policy_update_interval

        self.red_policies = []
        self.blue_policies = []

        self.add_current_policies(0.05, 0.05)
        self.red_policies.append(
            RedPolicy(self.collector.state_dict(), self.guide.state_dict(), self.counter, 0.05))

    def add_current_policies(self, rating_r, rating_b):
        self.blue_policies.append(BluePolicy(
            self.enemy.state_dict(), self.counter, rating_b))

    def sample(self):
        ratings_b = [p.rating for p in self.blue_policies]
        sum_b = sum(ratings_b)
        probs_b = [r/sum_b for r in ratings_b]

        index_b = np.random.choice(
            np.arange(len(self.blue_policies)), p=probs_b)

        self.enemy.load_state_dict(self.blue_policies[index_b].e_dict)
        print('New Enemy Policy, Rating: {} '.format(ratings_b[index_b]))

    def step(self, rating_r, rating_b):
        self.counter += 1

        if self.counter % self.policy_update_interval == 0:
            self.add_current_policies(rating_r, rating_b)
            self.sample()

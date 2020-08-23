import gym
from gym.wrappers import Monitor
import pyglet
import numpy as np
import matplotlib.pyplot as plt
import time

start = time.time()

env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
               red_guides=1, blue_collector=0)

num_eps = 64
max_timesteps = 500


for eps in range(num_eps):
    env.reset()
    for iter in range(max_timesteps):
        actions = np.random.randint(0, 4, size=2)

        obs, rewards, done, _ = env.step(actions)
        if done:
            break

print(time.time()-start)
env.close()

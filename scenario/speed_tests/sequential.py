import gym
from gym.wrappers import Monitor
import pyglet
import numpy as np
import matplotlib.pyplot as plt
import time

game_len = 500
env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
               game_length=game_len, competition=True)

num_eps = 40
max_timesteps = 1000

start = time.time()
for eps in range(num_eps):
    env.reset()
    for iter in range(max_timesteps):
        actions = np.random.randint(0, 4, size=3)

        obs, rewards, done, _ = env.step(actions)
        if done:
            break

print(time.time()-start)
env.close()

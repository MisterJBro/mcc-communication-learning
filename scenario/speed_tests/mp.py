from multiprocessing.dummy import Pool, freeze_support
import gym
import numpy as np
import time
from scenario.utils.envs import Envs

start = time.time()

num_eps = 1
max_timesteps = 500
envs = Envs(64)

for eps in range(num_eps):
    envs.reset()
    for iter in range(max_timesteps):
        actions = np.random.randint(0, 4, size=(64, 2))

        obs, rewards, done, _ = envs.step(actions)

print(time.time()-start)
envs.close()

from multiprocessing.dummy import Pool as ThreadPool
import gym
import numpy as np
import time


num_eps = 128
max_timesteps = 500
envs = Envs(4)

start = time.time()
for eps in range(num_eps):
    envs.reset()
    for iter in range(max_timesteps):
        actions = np.random.randint(0, 4, size=(4, 2))

        obs, rewards, done, _ = envs.step(actions)
        if done:
            break

print(time.time()-start)
envs.close()

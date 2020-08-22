from multiprocessing.dummy import Pool, freeze_support
import gym
import numpy as np
import time
from scenario.utils.envs import Envs


if __name__ == "__main__":
    freeze_support()
    num_eps = 40
    max_timesteps = 1000
    envs = Envs(8)
    pool = Pool(8)

    start = time.time()
    for eps in range(num_eps):
        envs.reset()
        for iter in range(max_timesteps):
            actions = np.random.randint(0, 4, size=(1, 2))

            obs, rewards, done, _ = envs.step(actions, pool)
            print(obs, rewards, done)
            if done:
                break

    print(time.time()-start)
    envs.close()

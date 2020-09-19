import numpy as np


def single_preprocess(obs):
    """ Processes a single observation into one hot encoding """
    # Both Player overlap each other
    x, y = np.where(obs == 5)
    if len(x) > 0:
        obs[x, y] = 3
    state = np.zeros((obs.size, 5), dtype=np.uint8)
    state[np.arange(obs.size), obs.reshape(-1)] = 1
    state = state.reshape(obs.shape + (5,))
    state = np.moveaxis(state, -1, 0)

    if len(x) > 0:
        state[4, x, y] = 1

    return state


def preprocess_with_score(obs, score):
    state = single_preprocess(obs)
    return np.concatenate([state.reshape(-1), score])


def preprocess(obs_list):
    """ Processes all observation """
    obs_c, obs_g, obs_e = [], [], []
    for obs in obs_list:
        obs_c.append(preprocess_with_score(
            obs[0][0], obs[0][1:]))
        obs_g.append(preprocess_with_score(
            obs[1][0], obs[1][1:]))
        obs_e.append(preprocess_with_score(
            obs[2][0], obs[2][1:]))
    return np.array([obs_c, obs_g, obs_e])


def preprocess_state(states):
    return [single_preprocess(s) for s in states]

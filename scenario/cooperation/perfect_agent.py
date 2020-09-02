import gym
import numpy as np

num_games = 10_000
max_steps = 500
total_rew = []

env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
               red_guides=1, blue_collector=0, seed=0)
collector = env.world.red_players[0]


def perfect_behavior(pos, trs):
    if pos[1] > trs and pos[0] == 1:
        return 3
    elif pos[1] < trs and pos[0] == 1:
        return 2
    elif pos[0] < 14 and pos[1] == trs:
        return 0
    else:
        return 1


for _ in range(num_games):
    obs = env.reset()

    game_rew = 0
    for step in range(max_steps):
        trs = env.world.treasure_tunnel
        pos = collector.pos

        obs, rew, _, _ = env.step([perfect_behavior(pos, trs), 4])
        game_rew += rew[0]

    total_rew.append(game_rew)

total_rew = np.array(total_rew)
mean_rew = np.round(total_rew.mean(), 3)

print('Perfect agent simulated {} games with avg reward of {:4} per game'.format(
    num_games, mean_rew))
env.close()

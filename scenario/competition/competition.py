import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from scenario.utils.buffer import Buffer
from scenario.utils.networks import Policy, ValueFunction
import seaborn as sns
import pathlib

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, env, seed=0, device='cuda:0', lr_policy=2e-3, lr_value=2e-3, gamma=0.99, max_steps=500,
                 hidden_size=128, batch_size=64, iters_policy=40, iters_value=40, lam=0.97, clip_ratio=0.2,
                 target_kl=0.03, num_layers=1, grad_clip=1.0, entropy_factor=0.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.width = 5
        self.height = 5
        self.env = env
        self.obs_dim = (self.num_world_blocks,) + env.observation_space.shape
        self.act_dim = env.action_space.nvec[0]
        self.agents_num = env.agents_num
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Network
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]
        self.device = torch.device(device)
        self.policy = Policy(
            in_dim+2, self.act_dim, rnn_hidden=hidden_size,  num_layers=num_layers).to(self.device)
        self.value = ValueFunction(
            in_dim+2, rnn_hidden=hidden_size,  num_layers=num_layers).to(self.device)
        self.trained_policy = Policy(
            in_dim, self.act_dim, rnn_hidden=hidden_size,  num_layers=num_layers).to(self.device)
        self.load_trained()

        self.optimizer_policy = optim.Adam(
            self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(
            self.value.parameters(), lr=lr_value)
        self.batch_size = batch_size
        self.iters_policy = iters_policy
        self.iters_value = iters_value
        self.criterion = nn.MSELoss()
        self.grad_clip = grad_clip

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.reset_state()

        # RL
        self.max_rew = 0
        self.gamma = gamma
        self.lam = lam
        self.max_steps = max_steps
        self.buffer = Buffer(self.max_steps*self.batch_size,
                             (in_dim+2,), self.gamma, self.lam)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.entropy_factor = entropy_factor

    def single_preprocess(self, obs):
        obs = obs[0]
        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        return np.moveaxis(state, -1, 0)

    def preprocess(self, obs):
        processed = self.single_preprocess(
            obs).reshape(-1).tolist() + [obs[1], obs[2]]

        return processed

    def preprocess_trained(self, obs):
        processed = self.single_preprocess(
            obs)
        tmp = np.copy(processed[4])
        processed[4] = np.copy(processed[3])
        processed[3] = tmp
        return processed

    def sample_batch(self):
        self.buffer.clear()
        rews = []

        while True:
            obs = self.env.reset()
            obs[0] = self.preprocess(obs[0])
            obs[1] = self.preprocess_trained(obs[1])
            episode_rew = 0
            for step in range(self.max_steps):
                act = self.get_action(obs)
                next_obs, rew, done, _ = self.env.step(act)
                next_obs[0] = self.preprocess(next_obs[0])
                next_obs[1] = self.preprocess_trained(next_obs[1])

                self.buffer.store(obs[0], act[0], rew[0])
                obs = next_obs
                episode_rew += rew[0]

                if done:
                    break
            rews.append(episode_rew)
            self.reward_and_advantage()
            self.reset_state()
            if self.buffer.ptr >= self.batch_size*self.max_steps:
                break

        return rews

    def reward_and_advantage(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[self.buffer.last_ptr:self.buffer.ptr], dtype=torch.float32).reshape(1, self.buffer.ptr-self.buffer.last_ptr, -1).to(self.device)
        with torch.no_grad():
            values = self.value(obs).cpu().numpy()
        self.buffer.expected_returns()
        self.buffer.advantage_estimation(values, 0.0)
        self.buffer.next_episode()

    def get_action(self, obs_list):
        obs = torch.as_tensor(
            obs_list[0], dtype=torch.float32).reshape(1, 1, -1).to(self.device)
        obs_trained = torch.as_tensor(
            obs_list[1], dtype=torch.float32).reshape(1, 1, -1).to(self.device)
        with torch.no_grad():
            dist, self.policy_state = self.policy.with_state(
                obs, self.policy_state)
            dist_trained, self.policy_state_trained = self.trained_policy.with_state(
                obs_trained, self.policy_state_trained)

        return [dist.sample().cpu().item(), dist_trained.sample().cpu().item()]

    def compute_policy_gradient(self, obs, act, adv, old_logp):
        dist = self.policy(obs)
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -torch.min(ratio*adv, clipped).mean() - \
            self.entropy_factor*dist.entropy().mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_policy(self, obs, act, adv):
        full_loss = 0
        with torch.no_grad():
            old_logp = self.policy(obs).log_prob(act).to(self.device)
        for i in range(self.iters_policy):
            self.optimizer_policy.zero_grad()
            loss, kl = self.compute_policy_gradient(obs, act, adv, old_logp)
            full_loss += loss.item()
            loss.backward()
            if kl > self.target_kl:
                return full_loss
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.grad_clip)
            self.optimizer_policy.step()
        return full_loss

    def update_value(self, obs, ret):
        full_loss = 0
        for i in range(self.iters_value):
            self.optimizer_value.zero_grad()
            input = self.value(obs)
            loss = self.criterion(input, ret)
            full_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value.parameters(), self.grad_clip)
            self.optimizer_value.step()
        return full_loss

    def update(self):
        obs = torch.as_tensor(
            self.buffer.obs_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)
        obs = obs.reshape(self.batch_size, self.max_steps, -1)
        act = torch.as_tensor(
            self.buffer.act_buf[:self.buffer.ptr], dtype=torch.int32, device=self.device)
        ret = torch.as_tensor(
            self.buffer.ret_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)
        self.buffer.standardize_adv()
        adv = torch.as_tensor(
            self.buffer.adv_buf[:self.buffer.ptr], dtype=torch.float32, device=self.device)

        pol_loss = self.update_policy(obs, act, adv)
        val_loss = self.update_value(obs, ret)
        return pol_loss, val_loss

    def train(self, epochs, prev_rews=[]):
        epoch_rews = []
        for epoch in range(epochs):
            rews = self.sample_batch()
            mean_rew = np.mean(rews)
            epoch_rews.append(mean_rew)
            if mean_rew > self.max_rew:
                self.max_rew = mean_rew
                self.save(epoch_rews)
            pol_loss, val_loss = self.update()

            print('Epoch: {:4}  Average Reward: {:6}  Policy Loss: {:7}  Value Loss: {:7}'.format(
                epoch, np.round(mean_rew, 3), np.round(pol_loss, 4), np.round(val_loss, 4)))

    def plot(self, arr, title='', xlabel='Epochs', ylabel='Average Reward'):
        sns.set()
        plt.plot(arr)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def save(self, rews=[], path='{}/model_comp.pt'.format(PROJECT_PATH)):
        torch.save({
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'optim_p': self.optimizer_policy.state_dict(),
            'optim_v': self.optimizer_value.state_dict(),
            'rews': rews
        }, path)

    def load(self, path='{}/model_comp.pt'.format(PROJECT_PATH)):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.optimizer_policy.load_state_dict(checkpoint['optim_p'])
        self.optimizer_value.load_state_dict(checkpoint['optim_v'])
        return checkpoint['rews']

    def load_trained(self, path='{}/model.pt'.format(PROJECT_PATH)):
        checkpoint = torch.load(path)
        self.trained_policy.load_state_dict(checkpoint['policy'])

    def test(self):
        obs = self.env.reset()
        episode_rew = 0

        while True:
            obs[0] = self.preprocess(obs[0])
            obs[1] = self.preprocess_trained(obs[1])
            self.env.render()
            act = self.get_action(obs)
            obs, rew, done, _ = self.env.step(act)

            episode_rew += rew[0]

            if done:
                break
        self.reset_state()

    def reset_state(self):
        self.policy_state = (
            torch.zeros(self.num_layers, 1, self.hidden_size,
                        device=self.device),
            torch.zeros(self.num_layers, 1, self.hidden_size,
                        device=self.device),
        )
        self.policy_state_trained = (
            torch.zeros(self.num_layers, 1, self.hidden_size,
                        device=self.device),
            torch.zeros(self.num_layers, 1, self.hidden_size,
                        device=self.device),
        )


if __name__ == "__main__":
    game_len = 500
    env = gym.make('gym_mcc_treasure_hunt:MCCTreasureHunt-v0',
                   red_guides=0, blue_collector=1, competition=True, game_length=game_len)
    agents = Agents(env, max_steps=game_len)
    agents.load()
    # agents.train(200)

    while True:
        input('Press enter to continue')
        agents.test()

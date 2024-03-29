import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
import random
import gym
import numpy as np
from scenario.competition.buffers import Buffers
from scenario.competition.networks import Normal
import seaborn as sns
import pathlib
from scenario.utils.envs import Envs

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, seed=0, device='cuda:0', lr_collector=1e-3, lr_enemy=1e-3, gamma=0.99, max_steps=500,
                 fc_hidden=64, rnn_hidden=128, batch_size=256, lam=0.97, clip_ratio=0.2, target_kl=0.01,
                 num_layers=1, grad_clip=1.0, symbol_num=5, tau=1.0, entropy_factor=-0.1):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.competition = False
        self.num_world_blocks = 5
        self.envs = Envs(batch_size, red_guides=0,
                         blue_collector=1, competition=self.competition)
        self.obs_dim = (self.num_world_blocks,) + \
            self.envs.observation_space.shape
        self.act_dim = self.envs.action_space.nvec[0]
        self.agents_num = self.envs.agents_num
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Networks
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]
        if self.competition:
            in_dim = in_dim+2
        self.device = torch.device(device)
        self.collector = Normal(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers).to(self.device)
        self.enemy = Normal(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers).to(self.device)
        self.enemy.load_state_dict(self.collector.state_dict())

        self.optimizer = optim.Adam(
            self.collector.parameters(), lr=lr_collector)
        milestones = [100, 200]
        self.scheduler = MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.5)
        self.batch_size = batch_size
        self.val_criterion = nn.MSELoss()
        self.pred_criterion = nn.CrossEntropyLoss()

        self.iters = 40
        self.red_iters = self.iters
        self.blue_iters = self.iters
        self.epochs_per_team = 5

        self.grad_clip = grad_clip
        self.symbol_num = symbol_num
        self.num_layers = num_layers
        self.rnn_hidden = rnn_hidden
        self.reset_states()

        # RL
        self.max_rew = 0
        self.gamma = gamma
        self.lam = lam
        self.max_steps = max_steps
        self.buffers = Buffers(self.batch_size, self.max_steps,
                               (in_dim,), self.gamma, self.lam, self.symbol_num)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def single_preprocess(self, obs, swap):
        """ Processes a single observation into one hot encoding """
        obs = obs.copy()
        x3, y3 = np.where(obs == 3)
        x4, y4 = np.where(obs == 4)
        x5, y5 = np.where(obs == 5)
        if len(x5) > 0:
            obs[x5, y5] = 3
        if swap:
            if len(x3) > 0:
                obs[x3, y3] = 4
            if len(x4) > 0:
                obs[x4, y4] = 3

        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        state = np.moveaxis(state, -1, 0)

        if len(x5) > 0:
            state[4, x5, y5] = 1
        return state

    def preprocess_with_score(self, obs, score, swap):
        state = self.single_preprocess(obs, swap)
        return np.concatenate([state.reshape(-1), score])

    def preprocess(self, obs_list):
        """ Processes all observation """
        obs_c, obs_e = [], []
        for obs in obs_list:
            if self.competition:
                obs_c.append(self.preprocess_with_score(
                    obs[0][0], obs[0][1:], False))
                obs_e.append(self.preprocess_with_score(
                    obs[1][0], obs[1][1:], True))
            else:
                obs_c.append(self.single_preprocess(
                    obs[0], False).reshape(-1))
                obs_e.append(self.single_preprocess(
                    obs[1], True).reshape(-1))
        return np.array([obs_c, obs_e])

    def sample_batch(self):
        """ Samples a batch of trajectories """
        self.buffers.clear()
        batch_rew = np.zeros((2, self.batch_size))
        obs = self.preprocess(self.envs.reset())

        for step in range(self.max_steps):
            acts = self.get_actions(obs)
            next_obs, rews, _, _ = self.envs.step(acts)
            next_obs = self.preprocess(next_obs)

            self.buffers.store(
                obs, acts, rews[:, 0], rews[:, 1])
            batch_rew[0] += rews[:, 0]
            batch_rew[1] += rews[:, 1]

            obs = next_obs
        self.reward_and_advantage()
        self.reset_states()

        return np.mean(batch_rew, 1)

    def reward_and_advantage(self):
        """ Calculates the rewards and General Advantage Estimation """
        obs_c = torch.as_tensor(self.buffers.buffer_c.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        obs_e = torch.as_tensor(self.buffers.buffer_e.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)

        with torch.no_grad():
            val_c = self.collector.value_only(obs_c).reshape(
                self.batch_size, self.max_steps).cpu().numpy()
            val_e = self.enemy.value_only(obs_e).reshape(
                self.batch_size, self.max_steps).cpu().numpy()

        self.buffers.expected_returns()
        self.buffers.advantage_estimation([val_c, val_e])
        self.buffers.standardize_adv()

    def get_actions(self, obs):
        """ Gets action according the agents networks """
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(
            self.agents_num, self.batch_size, 1, -1).to(self.device)

        act_dist_c, self.state_c = self.collector.next_action(
            obs[0], self.state_c)
        act_dist_e, self.state_e = self.enemy.next_action(
            obs[1], self.state_e)

        act_c = act_dist_c.sample().cpu().numpy()
        act_e = act_dist_e.sample().cpu().numpy()

        return np.stack([act_c, act_e]).T

    def compute_policy_gradient(self, net, dist, act, adv, old_logp):
        """ Computes the policy gradient with PPO """
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -(torch.min(ratio*adv, clipped)).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_net(self, net, opt, obs, act, adv, ret, iters):
        """ Updates the net """
        policy_loss = 0
        value_loss = 0
        other_done = False
        with torch.no_grad():
            old_logp = net.action_only(
                obs).log_prob(act).to(self.device)

        for i in range(iters):
            opt.zero_grad()

            dist, vals = net(obs)

            loss, kl = self.compute_policy_gradient(
                net, dist, act, adv, old_logp)
            policy_loss += loss.item()
            if kl > 0.03:
                return policy_loss, value_loss
            loss.backward(retain_graph=True)

            loss = self.val_criterion(vals.reshape(-1), ret)
            value_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net.parameters(), self.grad_clip)

            opt.step()
        return policy_loss, value_loss

    def update(self):
        """ Updates all nets """
        obs_c, act_c, rew_c, ret_c, adv_c, obs_e, act_e, ret_e, adv_e = self.buffers.get_tensors(
            self.device)

        # Training
        p_loss_c, v_loss_c = self.update_net(
            self.collector, self.optimizer, obs_c, act_c, adv_c, ret_c, self.red_iters)
        # p_loss_e, v_loss_e = self.update_net(
        #    self.enemy, self.optimizer_e, obs_e, act_e, adv_e, ret_e, self.blue_iters)

        self.scheduler.step()
        self.enemy.load_state_dict(self.collector.state_dict())

        trs_found = rew_c.nonzero().size(0)/self.batch_size

        return p_loss_c, v_loss_c, trs_found

    def train(self, epochs):
        """ Trains the agent for given epochs """
        epoch_rews = []

        for epoch in range(epochs):
            rew = self.sample_batch()
            epoch_rews.append(rew)

            if rew[0] > self.max_rew:
                self.max_rew = rew[0]
                self.save()
            p_loss_c, v_loss_c, trs_found = self.update()

            print('Epoch: {:4}  Collector Rew: {:4}  Enemy Rew: {:4}  Trs Found: {:3}'.format(
                epoch, np.round(rew[0], 3), np.round(rew[1], 3), np.round(trs_found, 1)))
        print(epoch_rews)

    def save(self, path='{}/model.pt'.format(PROJECT_PATH)):
        """ Saves the networks and optimizers to later continue training """
        torch.save({
            'collector': self.collector.state_dict(),
            'enemy': self.collector.state_dict(),
            'optim': self.optimizer.state_dict(),
        }, path)

    def load(self, path='{}/self_play.pt'.format(PROJECT_PATH)):
        """ Loads a training checkpoint """
        checkpoint = torch.load(path)
        self.collector.load_state_dict(checkpoint['collector'])
        self.enemy.load_state_dict(checkpoint['enemy'])
        self.optimizer.load_state_dict(checkpoint['optim'])

    def test(self):
        """ Tests the agent """
        obs = self.preprocess(self.envs.reset())
        episode_rew = 0
        msg_sum = np.zeros(self.symbol_num)

        for step in range(self.max_steps):
            import time
            time.sleep(0.05)
            self.envs.envs[0].render()

            acts = self.get_actions(obs)
            obs, rews, _, _ = self.envs.step(acts)
            obs = self.preprocess(obs)

            episode_rew += rews[0][0]
        print('Result reward: ', episode_rew)
        print(msg_sum)
        self.reset_states()

    def reset_states(self):
        """ Reset cell and hidden rnn states """
        self.state_c = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )
        self.state_e = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )


if __name__ == "__main__":
    agents = Agents()
    agents.load()
    # agents.train(300)

    import code
    # code.interact(local=locals())
    while True:
        input('Press enter to continue')
        agents.test()

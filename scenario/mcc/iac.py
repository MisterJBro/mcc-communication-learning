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
from scenario.mcc.buffers import Buffers
from scenario.mcc.networks import Policy, Listener, Speaker, Normal
import seaborn as sns
import pathlib
from scenario.utils.envs import Envs

PROJECT_PATH = pathlib.Path(
    __file__).parent.absolute().as_posix()


class Agents:
    def __init__(self, seed=0, device='cuda:0', lr_collector=1e-3, lr_guide=2e-3, lr_enemy=1e-3, gamma=0.99, max_steps=500,
                 fc_hidden=64, rnn_hidden=128, batch_size=256, lam=0.97, clip_ratio=0.2, target_kl=0.01,
                 num_layers=1, grad_clip=1.0, symbol_num=5, tau=1.0):
        # RNG seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Environment
        self.num_world_blocks = 5
        self.envs = Envs(batch_size, red_guides=1,
                         blue_collector=1, competition=True, show_messages=True)
        self.obs_dim = (self.num_world_blocks,) + \
            self.envs.observation_space.shape
        self.act_dim = self.envs.action_space.nvec[0]
        self.agents_num = self.envs.agents_num
        print('Observation shape:', self.obs_dim)
        print('Action number:', self.act_dim)
        print('Agent number:', self.agents_num)

        # Networks
        in_dim = self.obs_dim[0]*self.obs_dim[1]*self.obs_dim[2]+2
        self.device = torch.device(device)
        self.collector = Listener(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.guide = Speaker(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers, tau=tau).to(self.device)
        self.enemy = Normal(
            in_dim, self.act_dim, symbol_num, fc_hidden=fc_hidden, rnn_hidden=rnn_hidden, num_layers=num_layers).to(self.device)

        self.optimizer_c = optim.Adam(
            self.collector.parameters(), lr=lr_collector)
        self.optimizer_g = optim.Adam(
            self.guide.parameters(), lr=lr_guide)
        self.optimizer_e = optim.Adam(
            self.enemy.parameters(), lr=lr_enemy)
        milestones = [50, 300, 5000]
        self.scheduler_c = MultiStepLR(
            self.optimizer_c, milestones=milestones, gamma=0.1)
        self.scheduler_g = MultiStepLR(
            self.optimizer_g, milestones=milestones, gamma=0.1)
        self.scheduler_e = MultiStepLR(
            self.optimizer_e, milestones=milestones, gamma=0.1)
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
                               (in_dim,), self.act_dim, self.gamma, self.lam, self.symbol_num, (9,))
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl

    def single_preprocess(self, obs):
        """ Processes a single observation into one hot encoding """
        # Both Player overlap each other
        x, y = np.where(obs == 5)
        if len(x) > 0:
            obs[x, y] = 3
        state = np.zeros((obs.size, self.num_world_blocks), dtype=np.uint8)
        state[np.arange(obs.size), obs.reshape(-1)] = 1
        state = state.reshape(obs.shape + (self.num_world_blocks,))
        state = np.moveaxis(state, -1, 0)

        if len(x) > 0:
            state[4, x, y] = 1
        return state

    def preprocess_with_score(self, obs, score):
        state = self.single_preprocess(obs)
        return np.concatenate([state.reshape(-1), score])

    def preprocess(self, obs_list):
        """ Processes all observation """
        obs_c, obs_g, obs_e = [], [], []
        for obs in obs_list:
            obs_c.append(self.preprocess_with_score(
                obs[0][0], obs[0][1:]))
            obs_g.append(self.preprocess_with_score(
                obs[1][0], obs[1][1:]))
            obs_e.append(self.preprocess_with_score(
                obs[2][0], obs[2][1:]))
        return np.array([obs_c, obs_g, obs_e])

    def sample_batch(self):
        """ Samples a batch of trajectories """
        self.buffers.clear()
        batch_rew = np.zeros((3, self.batch_size))
        obs = self.preprocess(self.envs.reset())
        msg = torch.zeros((self.batch_size, self.symbol_num)).to(self.device)

        for step in range(self.max_steps):
            acts, dsts, next_msg = self.get_actions(obs, msg)
            next_obs, rews, _, states = self.envs.step(acts)
            next_obs = self.preprocess(next_obs)

            self.buffers.store(
                obs, acts, dsts, rews[:, 0].clip(min=0), rews[:, 1], rews[:, 2].clip(min=0), msg, states)
            batch_rew[0] += rews[:, 0]
            batch_rew[1] += rews[:, 1]
            batch_rew[2] += rews[:, 2]

            obs = next_obs
            msg = next_msg
        self.reward_and_advantage()
        self.reset_states()

        return np.mean(batch_rew, 1)

    def reward_and_advantage(self):
        """ Calculates the rewards and General Advantage Estimation """
        obs_c = torch.as_tensor(self.buffers.buffer_c.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        obs_g = torch.as_tensor(self.buffers.buffer_g.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        obs_e = torch.as_tensor(self.buffers.buffer_e.obs_buf, dtype=torch.float32).reshape(
            self.batch_size, self.max_steps, -1).to(self.device)
        msg = self.buffers.backprop_msg

        with torch.no_grad():
            val_c = self.collector.value_only(obs_c, msg).reshape(
                self.batch_size, self.max_steps).cpu().numpy()
            val_g = self.guide.value_only(obs_g).reshape(
                self.batch_size, self.max_steps).cpu().numpy()
            val_e = self.enemy.value_only(obs_e).reshape(
                self.batch_size, self.max_steps).cpu().numpy()

        self.buffers.expected_returns()
        self.buffers.advantage_estimation(
            [val_c, val_g, val_e])
        self.buffers.standardize_adv()

    def get_actions(self, obs, msg):
        """ Gets action according the agents networks """
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(
            self.agents_num, self.batch_size, 1, -1).to(self.device)
        msg = torch.as_tensor(msg, dtype=torch.float32).reshape(
            self.batch_size, 1, -1).to(self.device)

        act_dist_c, self.state_c = self.collector.next_action(
            obs[0], msg, self.state_c)
        act_dist_g, next_msg, self.state_g = self.guide.next_action(
            obs[1], self.state_g)
        act_dist_e, self.state_e = self.enemy.next_action(
            obs[2], self.state_e)

        dst_c = act_dist_c.probs.cpu().detach().numpy()
        dst_g = act_dist_g.probs.cpu().detach().numpy()
        dst_e = act_dist_e.probs.cpu().detach().numpy()

        act_c = act_dist_c.sample().cpu().numpy()
        act_g = act_dist_g.sample().cpu().numpy()
        act_e = act_dist_e.sample().cpu().numpy()

        return np.stack([act_c, act_g, act_e]).T, np.stack([dst_c, dst_g, dst_e]), next_msg

    def compute_policy_gradient(self, net, dist, act, adv, old_logp):
        """ Computes the policy gradient with PPO """
        logp = dist.log_prob(act)

        ratio = torch.exp(logp - old_logp)
        clipped = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio)*adv
        loss = -(torch.min(ratio*adv, clipped)).mean()
        kl_approx = (old_logp - logp).mean().item()
        return loss, kl_approx

    def update_net(self, net, opt, obs, act, adv, ret, iters, msg=None, other_net=None, other_obs=None, other_opt=None, other_act=None, other_adv=None, other_ret=None, enemy=False):
        """ Updates the net """
        policy_loss = 0
        value_loss = 0
        other_done = False
        with torch.no_grad():
            if msg is not None:
                old_logp = net.action_only(
                    obs, msg).log_prob(act).to(self.device)
            else:
                old_logp = net.action_only(
                    obs).log_prob(act).to(self.device)

            if other_net is not None:
                other_old_logp = other_net.action_only(
                    other_obs).log_prob(other_act).to(self.device)

        for i in range(iters):
            opt.zero_grad()
            if other_opt is not None:
                other_opt.zero_grad()

            if enemy:
                dist, vals = net(obs)
            elif msg is not None:
                dist, vals = net(obs, msg)
            else:
                dist, _, vals = net(obs)

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

            if other_net is not None and other_obs is not None:
                other_dist, _,  other_vals = other_net(other_obs)
                other_loss, other_kl = self.compute_policy_gradient(
                    other_net, other_dist, other_act, other_adv, other_old_logp)

                if other_kl > 0.01 and not other_done:
                    other_done = True
                else:
                    other_loss.backward(retain_graph=True)

                    other_loss = self.val_criterion(
                        other_vals.reshape(-1), other_ret)
                    other_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        other_net.parameters(), self.grad_clip)

            opt.step()
            if other_opt is not None:
                other_opt.step()
            if other_net is not None:
                _, msg, _ = other_net(other_obs[:, :-1])

                if other_done:
                    msg = msg.detach()

                msg = msg.reshape(self.batch_size, self.max_steps-1, -1)
                msg = torch.cat(
                    (torch.zeros(self.batch_size, 1, self.symbol_num).to(self.device), msg), 1)

        return policy_loss, value_loss

    def update(self):
        """ Updates all nets """
        obs_c, act_c, rew_c, ret_c, adv_c, dst_c, obs_g, act_g, ret_g, adv_g, dst_g, obs_e, act_e, rew_e, ret_e, adv_e, dst_e, msg, states = self.buffers.get_tensors(
            self.device)
        msg_ent = Categorical(
            probs=msg.reshape(-1, self.symbol_num).detach().cpu().mean(0)).entropy().item()

        # Training Collector/Msg/Guide - Collector - Guide
        _, _ = self.update_net(
            self.collector, self.optimizer_c, obs_c, act_c, adv_c, ret_c, 60, msg=msg, other_net=self.guide, other_obs=obs_g, other_opt=self.optimizer_g, other_act=act_g, other_adv=adv_g, other_ret=ret_g)
        p_loss_c, v_loss_c = self.update_net(
            self.collector, self.optimizer_c, obs_c, act_c, adv_c, ret_c, 0, msg=msg.detach())
        p_loss_g, v_loss_g = self.update_net(
            self.guide, self.optimizer_g, obs_g, act_g, adv_g, ret_g, 0)
        p_loss_e, v_loss_e = self.update_net(
            self.enemy, self.optimizer_e, obs_e, act_e, adv_e, ret_e, 10, enemy=True)

        if self.red_iters > 0:
            self.scheduler_c.step()
            self.scheduler_g.step()
        if self.blue_iters > 0:
            self.scheduler_e.step()

        trs_found = (rew_c.nonzero().size(0) +
                     rew_e.nonzero().size(0))/self.batch_size

        return p_loss_c, v_loss_c, p_loss_g, v_loss_g, p_loss_e, v_loss_e, msg_ent, trs_found

    def train(self, epochs):
        """ Trains the agent for given epochs """
        epoch_rews = []

        for epoch in range(epochs):
            rew = self.sample_batch()
            epoch_rews.append(rew)

            p_loss_c, v_loss_c, p_loss_g, v_loss_g, p_loss_e, v_loss_e, msg_ent, trs_found = self.update()

            blue = (trs_found-rew[0])/2
            red = trs_found - blue

            print('Epoch: {:3} Coll: {:4} Enemy: {:4} Guide: {:4} Msg {:4} Trs {:3} Red {:3} Blue {:3}'.format(
                epoch, np.round(rew[0], 3),  np.round(rew[2], 3), np.round(rew[1], 1), np.round(msg_ent, 3),  np.round(trs_found, 3), np.round(red, 2), np.round(blue, 2)))

            if red > self.max_rew:
                self.max_rew = red
                self.save()
        print(epoch_rews)

    def save(self, path='{}/model.pt'.format(PROJECT_PATH)):
        """ Saves the networks and optimizers to later continue training """
        torch.save({
            'collector': self.collector.state_dict(),
            'guide': self.guide.state_dict(),
            'enemy': self.enemy.state_dict(),
            'optim_c': self.optimizer_c.state_dict(),
            'optim_g': self.optimizer_g.state_dict(),
            'optim_e': self.optimizer_e.state_dict(),
        }, path)

    def load(self, path='{}/model.pt'.format(PROJECT_PATH)):
        """ Loads a training checkpoint """
        checkpoint = torch.load(path)
        self.collector.load_state_dict(checkpoint['collector'])
        self.guide.load_state_dict(checkpoint['guide'])
        self.enemy.load_state_dict(checkpoint['enemy'])
        self.optimizer_c.load_state_dict(checkpoint['optim_c'])
        self.optimizer_g.load_state_dict(checkpoint['optim_g'])
        self.optimizer_e.load_state_dict(checkpoint['optim_e'])

    def test(self):
        """ Tests the agent """
        obs = self.preprocess(self.envs.reset())
        msg = torch.zeros((self.batch_size, self.symbol_num)).to(self.device)
        episode_rew = 0
        msg_sum = np.zeros(self.symbol_num)
        self.already_in_first = False

        for step in range(self.max_steps):
            import time
            time.sleep(0.05)
            # msg[0] = torch.tensor([0., 0., 0., 1., 0.]).to(self.device)
            msg_show = msg[0].detach().cpu().numpy()
            self.last_message = self.interpret(msg_show)

            self.envs.envs[0].render(msg=self.interpret(msg_show))
            print(msg[0].detach().cpu().numpy())

            # print(self.envs.envs[0].world.red_)
            if self.envs.envs[0].world.red_players[0].pos[1] == 1 and self.envs.envs[0].world.red_players[0].pos[0] == 14:
                self.already_in_first = True

            if self.envs.envs[0].world.red_players[0].pos[1] == 19 and self.envs.envs[0].world.red_players[0].pos[0] == 14:
                self.already_in_first = False

            msg_sum += msg[0].detach().cpu().numpy()
            acts, _, msg = self.get_actions(obs, msg)
            obs, rews, _, _ = self.envs.step(acts)
            obs = self.preprocess(obs)

            episode_rew += rews[0][0]
            # print(rews[0])
        print('Result reward: ', episode_rew)
        print(msg_sum)
        self.reset_states()

    def interpret(self, msg):
        if msg[0]:
            return self.last_message
        if msg[1]:
            return 'Tunnel 2 or 4'
        if msg[2]:
            return 'Searching'
        if msg[3]:
            return 'Tunnel 2'
        if msg[4]:
            if self.already_in_first:
                return 'Tunnel 4'
            else:
                return 'Tunnel 1'

    def reset_states(self):
        """ Reset cell and hidden rnn states """
        self.state_c = (
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
            torch.zeros(self.num_layers, self.batch_size, self.rnn_hidden,
                        device=self.device),
        )
        self.state_g = (
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
    # agents.load()
    agents.train(500)

    import code
    # code.interact(local=locals())
    while True:
        input('Press enter to continue')
        agents.test()

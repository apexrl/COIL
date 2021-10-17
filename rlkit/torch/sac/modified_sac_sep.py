from collections import OrderedDict
import os
import traceback

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import rlkit.torch.pytorch_util as ptu
from rlkit.core.trainer import Trainer
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core import logger


class SoftActorCritic(Trainer):
    """
    version that:
        - uses reparameterization trick
        - has two Q functions and a V function
    """

    def __init__(
            self,
            policy,
            qfs,

            reward_scale=1.0,
            discount=0.99,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            lambda_lr=1e-3,
            soft_target_tau=1e-2,

            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,

            use_grad_clip=True,
            use_huber_loss=False,

            optimizer_class=optim.Adam,
            beta_1=0.9,
            q_lambda=2.0,
            q_lambda_min=0.01,
            q_lambda_max=10.0,
            target_thresh=40.0,

            q_update_times=1,
            bc_reg_weight=0.0
    ):
        self.policy = policy if isinstance(policy, list) else [policy]
        self.qfs = qfs if isinstance(qfs, list) else [qfs]

        self.reward_scale = reward_scale
        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight

        self.target_qfs = []
        for qf in self.qfs:
            self.target_qfs.append(qf.copy())
        self.eval_statistics = None

        self.use_grad_clip = use_grad_clip
        self.use_huber_loss = use_huber_loss
        assert not self.use_huber_loss

        self.policy_optimizer = []
        for pi in self.policy:
            self.policy_optimizer.append(optimizer_class(
                pi.parameters(),
                lr=policy_lr,
                betas=(beta_1, 0.999)
            ))
        self.qfs_optimizer = []
        for qf in self.qfs:
            self.qfs_optimizer.append(optimizer_class(
                qf.parameters(),
                lr=qf_lr,
                betas=(beta_1, 0.999)
            ))

        self.log_lambda = ptu.zeros(1, requires_grad=True)
        self.lambda_optimizer = optimizer_class(
            [self.log_lambda],
            lr=lambda_lr,
            betas=(beta_1, 0.999)
        )
        self.lambda_min = q_lambda_min
        self.lambda_max = q_lambda_max
        self.ac_extra_loss = None

        self.target_thresh = target_thresh
        print("TARGET THRESH = ", target_thresh)

        self.q_update_times = q_update_times
        print("Q UPDATE TIMES = ", q_update_times)
        self.bc_reg_weight = bc_reg_weight
        print("BC REG WEIGHT = ", bc_reg_weight)

    def q_min(self, obs, actions):
        qfs_new_actions = [qf(obs, actions) for qf in self.qfs]
        qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
        qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)
        return qfs_new_actions

    def policy_mean(self, obs, return_log_prob=False):
        policy_outputs = [pi(obs, return_log_prob=return_log_prob) for pi in self.policy]
        next_actions = [po[0] for po in policy_outputs]
        next_actions = torch.stack(next_actions)
        next_actions = torch.mean(next_actions, dim=0)
        return next_actions

    def train_q_net(self, batch, **kwargs):
        q_id = kwargs.get("q_id", None)
        q_loss = None
        q_target_loss = None
        q_extra_loss = None
        qfs_pred = None

        if q_id is None:
            return None, None, None, None

        rewards = self.reward_scale * batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        if self.q_update_times == -1:
            q_update_times = 1 + kwargs.get("extra_upd", 0)
        else:
            q_update_times = self.q_update_times if kwargs.get("extra_upd", False) else 1
        assert q_update_times >= 1
        for _ in range(q_update_times):
            with torch.no_grad():
                next_actions = self.policy_mean(next_obs)
                target_q_value = self.target_qfs[q_id](next_obs, next_actions)
                q_target = rewards + (1. - terminals) * self.discount * target_q_value

            with torch.no_grad():
                _actions = self.policy_mean(obs)
                _q_target = self.target_qfs[q_id](obs, _actions)

            qfs_pred = self.qfs[q_id](obs, actions)
            _qfs_pred = self.qfs[q_id](obs, _actions)

            q_target_loss = 0.5 * torch.mean((qfs_pred - q_target.detach()) ** 2)
            q_extra_loss = self.log_lambda.exp() * torch.mean(
                (_qfs_pred - _q_target.detach()) ** 2 - self.target_thresh)
            q_loss = q_target_loss + q_extra_loss

            opt = self.qfs_optimizer[q_id]
            opt.zero_grad()
            q_loss.backward(retain_graph=True)
            clip_gradient(opt)
            opt.step()

        return q_loss, q_target_loss, q_extra_loss, qfs_pred

    def train_lambda(self, q_extra_loss, update_lambda):
        if not update_lambda:
            return None

        lambda_loss = ptu.zeros(1, requires_grad=True) - q_extra_loss

        self.lambda_optimizer.zero_grad()
        lambda_loss.backward()
        self.lambda_optimizer.step()
        self.log_lambda.data.clamp_(min=np.log(self.lambda_min), max=np.log(self.lambda_max))
        return lambda_loss

    def train_policy_rl(self, batch, p_id):
        if p_id is None:
            return None

        obs = batch['observations']
        actions = batch['actions']

        log_prob = self.policy[p_id].get_log_prob(obs, actions)
        mle_policy_loss = -1.0 * log_prob.mean()

        action_num = 10
        obs = obs.unsqueeze(1).repeat(1, action_num, 1).view(-1, obs.shape[1])

        policy_outputs = self.policy[p_id](obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        qfs_new_actions = self.q_min(obs, new_actions)

        policy_grad_loss = torch.mean(-qfs_new_actions)
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        bc_reg_loss = self.bc_reg_weight * mle_policy_loss
        policy_reg_loss = mean_reg_loss + std_reg_loss + bc_reg_loss
        policy_loss = policy_grad_loss + policy_reg_loss

        opt = self.policy_optimizer[p_id]
        opt.zero_grad()
        policy_loss.backward()
        clip_gradient(opt)
        opt.step()

        return policy_loss

    def train_policy_bc(self, batch, p_id):
        if p_id is None:
            return

        obs = batch["observations"]
        acts = batch["actions"]

        log_prob = self.policy[p_id].get_log_prob(obs, acts)
        policy_loss = -1.0 * log_prob.mean()

        policy_outputs = self.policy[p_id](obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer[p_id].zero_grad()
        policy_loss.backward()
        self.policy_optimizer[p_id].step()

    def train_step(self, batch, **kwargs):
        training_mode = kwargs.get("training_mode", None)
        q_id = kwargs.get("q_id", None)
        p_id = kwargs.get("p_id", None)
        update_lambda = kwargs.get("update_lambda", False)

        if training_mode is None or training_mode == "rl":
            q_loss, q_target_loss, q_extra_loss, q_pred = self.train_q_net(batch, **kwargs)
            if q_extra_loss is not None:
                if self.ac_extra_loss is None:
                    self.ac_extra_loss = q_extra_loss
                else:
                    self.ac_extra_loss = self.ac_extra_loss + q_extra_loss
            lambda_loss = self.train_lambda(self.ac_extra_loss, update_lambda)
            policy_loss = self.train_policy_rl(batch, p_id)

            if self.eval_statistics is None:
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                self.eval_statistics = OrderedDict()
                if q_id is not None:
                    self.eval_statistics['Reward Scale'] = self.reward_scale
                    self.eval_statistics['QFs Loss'] = ptu.get_numpy(q_loss)
                    self.eval_statistics['QFs Target Loss'] = ptu.get_numpy(q_target_loss)
                    self.eval_statistics['QFs Extra Loss'] = ptu.get_numpy(q_extra_loss)
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Q%d Predictions' % (q_id + 1),
                        ptu.get_numpy(q_pred),
                    ))
                    self.eval_statistics['Lambda'] = ptu.get_numpy(self.log_lambda.exp())
                if update_lambda:
                    self.eval_statistics['Lambda Loss'] = np.mean(ptu.get_numpy(lambda_loss))
                if p_id is not None:
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Policy%d Loss' % (p_id + 1),
                        np.mean(ptu.get_numpy(policy_loss)),
                    ))

        elif training_mode == "bc":
            try:
                p_id = kwargs.get("p_id", None)
                self.train_policy_bc(batch, p_id)
            except Exception as e:
                traceback.print_exc()
                raise e

        else:
            raise ValueError("Invalid training mode")

    @property
    def networks(self):
        return self.policy + self.qfs + self.target_qfs

    def _update_target_network(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            ptu.soft_update_from_to(qf, target_qf, self.soft_target_tau)

    def end_single_loop(self):
        self._update_target_network()
        self.ac_extra_loss = None

    def get_snapshot(self):
        return dict(
            qfs=self.qfs,
            target_qfs=self.target_qfs,
            policy=self.policy
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        print("END_EPOCH")
        self.eval_statistics = None


def clip_gradient(optimizer, grad_clip=0.5):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def huber_loss(x, delta=1):
    return torch.where(
        torch.abs(x) < delta,
        0.5 * (x ** 2),
        delta * (torch.abs(x) - 0.5 * delta)
    )

from collections import OrderedDict
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

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
            vf,

            reward_scale=1.0,
            discount=0.99,

            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            soft_target_tau=1e-2,

            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,

            use_grad_clip=True,

            optimizer_class=optim.Adam,
            beta_1=0.9,
            q_lambda=2.0
    ):
        self.policy = policy
        self.qfs = qfs if isinstance(qfs, list) else [qfs]
        self.vf = vf
        print("NUM Q NET = ", len(self.qfs))

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

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )
        self.qfs_optimizer = []
        for qf in self.qfs:
            self.qfs_optimizer.append(optimizer_class(
                qf.parameters(),
                lr=qf_lr,
                betas=(beta_1, 0.999)
            ))
        print("Q LR = ", qf_lr)
        self._lambda = q_lambda
        print("Q LAMBDA = ", q_lambda)

    def train_step(self, batch, **kwargs):
        training_mode = kwargs["training_mode"]
        if training_mode is None or training_mode == "sac":
            rewards = self.reward_scale * batch['rewards']
            terminals = batch['terminals']
            obs = batch['observations']
            actions = batch['actions']
            next_obs = batch['next_observations']

            # torch.autograd.set_detect_anomaly(True)

            """
            QF Loss
            """
            with torch.no_grad():
                policy_outputs_ = self.policy(next_obs, return_log_prob=True)
                next_actions = policy_outputs_[0].detach()
                target_q_values = [qf(next_obs, next_actions) for qf in self.target_qfs]
                target_q_values = torch.cat(target_q_values, dim=-1)
                min_q_values = torch.min(target_q_values, dim=-1).values
                target_v_values = torch.unsqueeze(min_q_values, dim=-1)
                q_target = rewards + (1. - terminals) * self.discount * target_v_values

            with torch.no_grad():
                _policy_outputs = self.policy(obs, return_log_prob=True)
                _actions = _policy_outputs[0].detach()
                _target_q_values = [qf(obs, _actions) for qf in self.target_qfs]
                _target_q_values = torch.cat(_target_q_values, dim=-1)
                _min_q_values = torch.min(_target_q_values, dim=-1).values
                _q_target = torch.unsqueeze(_min_q_values, dim=-1)

            qfs_pred = [qf(obs, actions) for qf in self.qfs]
            _qfs_pred = [qf(obs, _actions) for qf in self.qfs]

            qfs_loss = [0.5 * torch.mean((q_pred - q_target.detach()) ** 2)
                        + self._lambda * torch.mean((_q_pred - _q_target.detach()) ** 2)
                        for q_pred, _q_pred in zip(qfs_pred, _qfs_pred)]

            '''qfs_loss = [0.5 * torch.mean((q_pred - q_target.detach()) ** 2)
                        for q_pred in qfs_pred]'''

            q_loss_mean = .0
            for q_loss in qfs_loss:
                q_loss_mean += np.mean(ptu.get_numpy(q_loss))
            q_loss_mean /= len(qfs_loss)

            for i, opt in enumerate(self.qfs_optimizer):
                opt.zero_grad()
                qfs_loss[i].backward()
                clip_gradient(opt)
                opt.step()

            '''
            target_v_values = self.target_vf(next_obs)
            q_target = rewards + (1. - terminals) * self.discount * target_v_values
            qf1_loss = 0.5 * torch.mean((q1_pred - q_target.detach()) ** 2)
            qf2_loss = 0.5 * torch.mean((q2_pred - q_target.detach()) ** 2)
            
            """
            VF Loss
            """
            q1_new_acts = self.qf1(obs, new_actions)
            q2_new_acts = self.qf2(obs, new_actions)
            q_new_actions = torch.min(q1_new_acts, q2_new_acts)

            q_new_actions = q_new_actions.view(batch['actions'].shape[0], action_num, 1)
            q_new_actions, ind = torch.min(q_new_actions, dim=1)
            log_pi = log_pi.view(batch['actions'].shape[0], action_num)
            log_pi = torch.gather(log_pi, 1, ind)
            policy_log_std = policy_log_std.view(batch['actions'].shape[0], action_num, batch['actions'].shape[1])
            ind = ind.unsqueeze(2).repeat(1, 1, batch['actions'].shape[1])
            policy_log_std = torch.gather(policy_log_std, 1, ind)
            policy_mean = policy_mean.view(batch['actions'].shape[0], action_num, batch['actions'].shape[1])
            policy_mean = torch.gather(policy_mean, 1, ind)

            v_target = q_new_actions - log_pi
            vf_loss = 0.5 * torch.mean((v_pred - v_target.detach()) ** 2)
            '''
            """
            Policy Loss
            """

            # Make sure policy accounts for squashing functions like tanh correctly!
            action_num = 10
            obs = obs.unsqueeze(1).repeat(1, action_num, 1).view(-1, obs.shape[1])

            policy_outputs = self.policy(obs, return_log_prob=True)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
            qfs_new_actions = [qf(obs, new_actions) for qf in self.qfs]
            qfs_new_actions = torch.cat(qfs_new_actions, dim=-1)
            qfs_new_actions = torch.unsqueeze(torch.min(qfs_new_actions, dim=-1).values, dim=-1)

            policy_grad_loss = torch.mean(-qfs_new_actions)
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss
            policy_loss = policy_grad_loss + policy_reg_loss

            """
            Update networks
            """

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            clip_gradient(self.policy_optimizer)
            self.policy_optimizer.step()

            self._update_target_network()

            """
            Save some statistics for eval
            """
            if self.eval_statistics is None:
                """
                Eval should set this to None.
                This way, these statistics are only computed for one batch.
                """
                self.eval_statistics = OrderedDict()
                self.eval_statistics['Reward Scale'] = self.reward_scale
                self.eval_statistics['QFs Loss'] = q_loss_mean
                self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                    policy_loss
                ))
                for i, q_pred in enumerate(qfs_pred):
                    self.eval_statistics.update(create_stats_ordered_dict(
                        'Q%d Predictions' % (i + 1),
                        ptu.get_numpy(q_pred),
                    ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))

        elif training_mode == "bc":

            '''obs = batch["observations"]
            acts = batch["actions"]

            log_prob = self.policy.get_log_prob(obs, acts)
            policy_loss = -1.0 * log_prob.mean()

            policy_outputs = self.policy(obs, return_log_prob=True)
            new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss
            policy_loss = policy_loss + policy_reg_loss

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()'''

        else:
            raise ValueError("Invalid training mode")

    @property
    def networks(self):
        return [self.policy] + self.qfs + self.target_qfs

    def _update_target_network(self):
        for qf, target_qf in zip(self.qfs, self.target_qfs):
            ptu.soft_update_from_to(qf, target_qf, self.soft_target_tau)

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

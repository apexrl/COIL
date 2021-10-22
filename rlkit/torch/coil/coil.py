import torch
import torch.optim as optim
from collections import OrderedDict

from rlkit.core.trainer import Trainer
import rlkit.torch.pytorch_util as ptu


class COIL(Trainer):
    def __init__(
            self,
            policy,
            policy_lr=1e-3,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            optimizer_class=optim.Adam,
            beta_1=0.9,

            **kwargs
    ):
        self.policy = policy
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            betas=(beta_1, 0.999)
        )

        self.use_l2 = kwargs.get("use_l2", True)
        print("Use L2 norm: ", self.use_l2)
        self.bc_mode = kwargs.get("bc_mode", "mle")
        print("BC mode: ", self.bc_mode)
        assert self.bc_mode == "mse" or self.bc_mode == "mle", "Invalid BC mode!"

        self.eval_statistics = None

    def train_step(self, batch):
        obs = batch['observations']
        acts = batch['actions']
        policy_outputs = self.policy(obs, return_log_prob=True)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        if self.bc_mode == "mle":
            log_prob = self.policy.get_log_prob(obs, acts)
            policy_loss = -1.0 * log_prob.mean()
        elif self.bc_mode == "mse":
            squared_diff = (new_actions - acts) ** 2
            policy_loss = torch.sum(squared_diff, dim=-1).mean()
        else:
            raise ValueError("Invalid BC mode!")

        if self.use_l2:
            mean_reg_loss = self.policy_mean_reg_weight * (policy_mean ** 2).mean()
            std_reg_loss = self.policy_std_reg_weight * (policy_log_std ** 2).mean()
            policy_reg_loss = mean_reg_loss + std_reg_loss
        else:
            policy_reg_loss = 0
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics["Policy Loss"] = ptu.get_numpy(policy_loss)

    @property
    def networks(self):
        return [self.policy]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
        )

    def get_eval_statistics(self):
        return self.eval_statistics

    def end_epoch(self):
        print("END_EPOCH")
        self.eval_statistics = None

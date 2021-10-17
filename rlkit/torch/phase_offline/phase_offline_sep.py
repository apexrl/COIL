import numpy as np
from collections import OrderedDict
import torch
import gtimer as gt
import traceback

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import eval_util
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core import logger


class PhaseOffline(TorchBaseAlgorithm):
    """
        Phase Offline Reinforcement Learning.
    """
    def __init__(
        self,

        mode,
        policy_trainer,

        replay_buffer,
        random_reward=100,
        ratio=0.7,
        pretrain_times=10,

        clamp_magnitude=1.0,

        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_policy_update_loops_per_train_call=100,
        num_bc_update_loops_per_train_call=100,
        num_policy_updates_per_loop_iter=1,

        use_grad_pen=True,
        grad_pen_weight=10,

        filter_type=None,
        filter_tau=None,

        **kwargs
    ):
        assert mode in ['reward', 'occupancy'], 'Invalid  algorithm!'
        if kwargs['wrap_absorbing']:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        self.ratio = self.epsilon = ratio
        self.random_reward = random_reward
        self.pretrain_times = pretrain_times

        self.replay_buffer = replay_buffer
        self.pair_replay_buffer = None

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.use_grad_pen = use_grad_pen
        self.grad_pen_weight = grad_pen_weight

        self.num_policy_update_loops_per_train_call = num_policy_update_loops_per_train_call
        self.num_bc_update_loops_per_train_call = num_bc_update_loops_per_train_call
        self.num_policy_updates_per_loop_iter = num_policy_updates_per_loop_iter

        self.trained = False
        self.last_is_bc = 0

        self.total_train_steps = 0
        self.used_traj_orders = []
        self.used_traj_rews = []
        self.bc_per_iter = 0
        self.min_traj_reward = []
        self.filter_type = filter_type
        if self.filter_type == "running_mean":
            self.running_mean = 0
            self.filter_tau = filter_tau if filter_tau else 0.99

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        for i in range(self.pretrain_times):
            self._do_training(-1, pretrain=True) # Actually not pretrained
        self.trained = True

    def start_training(self, start_epoch=0):
        observation = self._start_new_rollout()
        out_of_data = False

        for epoch in gt.timed_for(
            range(start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            try:
                self._start_epoch(epoch)
                for steps_this_epoch in range(self.num_env_steps_per_epoch):
                    self._n_env_steps_total += 1
                    if self._n_env_steps_total % self.num_steps_between_train_calls == 0:
                        gt.stamp('sample')
                        self._try_to_train(epoch)
                        gt.stamp('train')
            except ValueError:
                out_of_data = True
                break
            finally:
                gt.stamp('sample')
                self._try_to_eval(epoch, out_of_data=out_of_data)
                gt.stamp('eval')
                self._end_epoch()
        
    def _can_train(self):
        return self.replay_buffer.get_size() > 0

    def _try_to_train(self, epoch):
        if self._can_train():
            self.training_mode(True)
            self._do_training(epoch)
            self._n_train_steps_total += 1
            self.training_mode(False)
        pass
    
    def get_batch(self, batch_size, keys=None):
        batch = self.pair_replay_buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)

        return batch

    def _end_epoch(self):
        """
        if len(self.min_traj_reward) == 0:
            self.min_traj_reward.append(-np.inf)
        self.replay_buffer.filter_below(np.min(self.min_traj_reward))
        self.min_traj_reward[:] = []
        """
        if self.filter_type and len(self.min_traj_reward) > 0:
            if self.filter_type == "mean":
                self.replay_buffer.filter_below(np.nanmean(self.min_traj_reward))
            elif self.filter_type == "running_mean":
                nan_min = np.nanmin(self.min_traj_reward)
                if not np.isnan(nan_min):
                    self.running_mean = self.filter_tau * self.running_mean \
                                        + (1 - self.filter_tau) * nan_min
                    self.replay_buffer.filter_below(self.running_mean)
                self.min_traj_reward = []
        self.policy_trainer.end_epoch()
        super()._end_epoch()

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return self.trained
    '''
    def evaluate(self, epoch):
        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        evaluate(epoch)
    '''

    def _do_training(self, epoch, pretrain=False):
        if not pretrain:
            self.pair_replay_buffer, training_mode, traj_steps, traj_order, traj_rews, min_reward = \
                self.replay_buffer.get_sufficient_buffer(self.epsilon, self.policy_trainer.policy)
            self.total_train_steps += traj_steps
            self.used_traj_orders.append(traj_order)
            self.used_traj_rews.append(traj_rews)
            self.bc_per_iter += int(training_mode == "bc")
            self.min_traj_reward.append(min_reward)
        else:
            return

        num_loops_per_train_call = self.num_policy_update_loops_per_train_call if training_mode == "rl" \
            else self.num_bc_update_loops_per_train_call

        for _ in range(num_loops_per_train_call):
            try:
                for idx in range(len(self.policy_trainer.qfs)):
                    self._do_policy_training(epoch, pretrain, training_mode=training_mode, q_id=idx,
                                             extra_upd=self.last_is_bc)

                self._do_policy_training(epoch, pretrain, training_mode=training_mode, update_lambda=True)

                for idx in range(len(self.policy_trainer.policy)):
                    self._do_policy_training(epoch, pretrain, training_mode=training_mode, p_id=idx)

                if training_mode is None or training_mode == "rl":
                    self.policy_trainer.end_single_loop()
            except Exception as e:
                traceback.print_exc()
                raise e

        if not pretrain:
            self.last_is_bc = (self.last_is_bc + 1) if training_mode == "bc" else 0

    def _do_policy_training(self, epoch, pretrain=False, **kwargs):
        policy_batch = self.get_batch(self.policy_optim_batch_size)
        # policy optimization step
        self.policy_trainer.train_step(policy_batch, **kwargs)

    def _handle_step(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        absorbing,
        agent_info,
        env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            absorbing=absorbing,
            agent_infos=agent_info,
            env_infos=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder
            )
            self._current_path_builder = PathBuilder()

    @property
    def networks(self):
        return self.exploration_policy + self.policy_trainer.networks

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        snapshot.update(self.policy_trainer.get_snapshot())
        return snapshot

    def to(self, device):
        super().to(device)

    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        self.eval_statistics = OrderedDict()
        if self.policy_trainer.get_eval_statistics() is not None:
            self.eval_statistics.update(self.policy_trainer.get_eval_statistics())
        statistics = OrderedDict()
        try:
            statistics.update(self.eval_statistics)
            self.eval_statistics = None
        except:
            print('No Stats to Eval')

        logger.log("Collecting samples for evaluation")
        test_paths = self.eval_sampler.obtain_samples()

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths)
        if hasattr(self.env, "log_statistics"):
            statistics.update(self.env.log_statistics(test_paths))
        if epoch % self.freq_log_visuals == 0:
            if hasattr(self.env, "log_visuals"):
                self.env.log_visuals(test_paths, epoch, logger.get_snapshot_dir())
        
        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns

        statistics['Total Training Steps'] = self.total_train_steps
        statistics['Used Traj Orders'] = self.used_traj_orders
        self.used_traj_orders = []
        statistics['Used Traj Rewards'] = self.used_traj_rews
        self.used_traj_rews = []
        statistics['BC per Iteration'] = self.bc_per_iter
        self.bc_per_iter = 0
        if self.filter_type == "running_mean":
            statistics['Filter'] = self.running_mean

        for key, value in statistics.items():
            logger.record_tabular(key, value)
        
        best_statistic = statistics[self.best_key]
        if best_statistic > self.best_statistic_so_far:
            self.best_statistic_so_far = best_statistic
            if self.save_best and epoch >= self.save_best_starting_from_epoch:
                data_to_save = {
                    'epoch': epoch,
                    'statistics': statistics
                }
                data_to_save.update(self.get_epoch_snapshot(epoch))
                logger.save_extra_data(data_to_save, 'best.pkl')
                print('\n\nSAVED BEST\n\n')

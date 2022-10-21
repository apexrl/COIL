import traceback

import numpy as np
from collections import OrderedDict
import gtimer as gt

from rlkit.torch.core import np_to_pytorch_batch
from rlkit.core import eval_util, logger
from rlkit.torch.torch_base_algorithm import TorchBaseAlgorithm
from rlkit.data_management.path_builder import PathBuilder
from rlkit.data_management.episodic_replay_buffer_coil import EpisodicReplayBuffer
from rlkit.torch.coil.coil import COIL


class PhaseOffline(TorchBaseAlgorithm):
    """
        Phase Offline Reinforcement Learning.
    """
    def __init__(
        self,

        mode,
        policy_trainer: COIL,

        replay_buffer: EpisodicReplayBuffer,
        min_reward=100,
        pretrain_times=0,
        pretrain_num=-1,

        policy_optim_batch_size=1024,
        policy_optim_batch_size_from_expert=0,

        num_bc_update_loops_per_train_call=100,

        filter_type=None,
        filter_tau=None,

        **kwargs
    ):
        assert mode in ['reward', 'occupancy'], 'Invalid  algorithm!'
        if kwargs['wrap_absorbing']:
            raise NotImplementedError()
        super().__init__(**kwargs)

        self.mode = mode
        if mode == "occupancy":
            self.best_key = "SuccessRate"
        self.pretrain_times = pretrain_times
        self.pretrain_num = pretrain_num

        self.replay_buffer = replay_buffer
        self.pair_replay_buffer = None

        self.policy_trainer = policy_trainer
        self.policy_optim_batch_size = policy_optim_batch_size
        self.policy_optim_batch_size_from_expert = policy_optim_batch_size_from_expert

        self.num_bc_update_loops_per_train_call = num_bc_update_loops_per_train_call

        self.trained = False

        self.total_train_steps = 0
        self.used_traj_orders = []
        self.used_traj_rews = []
        self.bc_per_iter = 0
        self.min_traj_reward = []
        self.filter_type = filter_type
        if self.filter_type == "running_mean":
            self.running_mean = min(0, min_reward)
            if self.mode == "occupancy":
                self.running_mean = -5
            self.filter_tau = filter_tau if filter_tau else 0.99
        self.shrink_buffer = kwargs.get("shrink_buffer", False)
        self.shrink_count = 0

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        if self.pretrain_times > 0:
            self._do_training(-1, pretrain=True)
        self.trained = True

    def start_training(self, start_epoch=0):
        # observation = self._start_new_rollout()
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
    
    def get_batch(self, batch_size, keys=None):
        batch = self.pair_replay_buffer.random_batch(batch_size, keys=keys)
        batch = np_to_pytorch_batch(batch)
        return batch

    def _start_epoch(self, epoch):
        super()._start_epoch(epoch)
        if self.shrink_buffer and epoch % 20 == 0:
            self.replay_buffer.shrink_buffer()

    def _end_epoch(self):
        if self.filter_type and len(self.min_traj_reward) > 0:
            self._filter_buffer()
        self.policy_trainer.end_epoch()
        super()._end_epoch()

    def _filter_buffer(self):
        if self.filter_type == "mean":
            self.replay_buffer.filter_below(np.nanmean(self.min_traj_reward))
        elif self.filter_type == "running_mean":
            nan_min = np.nanmin(self.min_traj_reward)
            if not np.isnan(nan_min):
                self.running_mean = self.filter_tau * self.running_mean \
                                    + (1 - self.filter_tau) * nan_min
                self.replay_buffer.filter_below(self.running_mean)
            self.min_traj_reward = []

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

    def _do_training(self, epoch, pretrain=False):
        if not pretrain:
            try:
                self.pair_replay_buffer, training_mode, traj_steps, traj_order, traj_rews, min_reward = \
                    self.replay_buffer.get_sufficient_buffer(policy_func=self.policy_trainer.policy, shrink_buffer=self.shrink_buffer)
                assert training_mode == "bc"
                self.total_train_steps += traj_steps
                self.used_traj_orders.append(traj_order)
                self.used_traj_rews.append(traj_rews)
                self.min_traj_reward.append(min_reward)
            except Exception as e:
                traceback.print_exc()
                raise e
        else:
            self.pair_replay_buffer = self.replay_buffer.get_pretrain_buffer(self.pretrain_num)
            if self.pair_replay_buffer.get_traj_num() == 0:
                return
            logger.log("Pretrain %d times." % self.pretrain_times)
            training_mode = "pretrain"

        num_loops_per_train_call = self.num_bc_update_loops_per_train_call if training_mode == "bc" \
            else self.pretrain_times

        for loop in range(num_loops_per_train_call):
            self._do_policy_training()

    def _do_policy_training(self):
        policy_batch = self.get_batch(self.policy_optim_batch_size)
        # policy optimization step
        self.policy_trainer.train_step(policy_batch)

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
        return [self.exploration_policy] + self.policy_trainer.networks

    def get_epoch_snapshot(self, epoch, save_all=False):
        snapshot = super().get_epoch_snapshot(epoch, save_all)
        snapshot.update(self.policy_trainer.get_snapshot())
        snapshot["filter"] = self.running_mean
        snapshot["filter_tau"] = self.filter_tau
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
        
        if self.mode == "reward":
            average_returns = eval_util.get_average_returns(test_paths)
            statistics['AverageReturn'] = average_returns
        elif self.mode == "occupancy":
            success_rate = eval_util.get_success_rate(test_paths)
            statistics['SuccessRate'] = success_rate

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

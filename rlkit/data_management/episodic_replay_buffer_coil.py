import numpy as np
import gc
import heapq
import torch

from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger


class EpisodicReplayBuffer(ReplayBuffer):
    """
    A class used to save and replay data.
    """

    def __init__(
            self,
            max_replay_buffer_size,
            max_sub_buf_size,
            observation_dim,
            action_dim,
            random_seed=1995,
            gamma=0.99,

            bc_sampling_with_rep=True,
            traj_sel_crt="alpha2",
            bc_traj_limit=2,

            only_bc=False,
            data_aug=1,
            **kwargs
    ):
        self._random_seed = random_seed
        self._np_rand_state = np.random.RandomState(random_seed)
        self.gamma = gamma

        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._max_sub_buf_size = max_sub_buf_size

        if isinstance(observation_dim, tuple):
            dims = [d for d in observation_dim]
            dims = [max_replay_buffer_size] + dims
            dims = tuple(dims)
            self._observations = np.zeros(dims)
            self._next_obs = np.zeros(dims)
        elif isinstance(observation_dim, dict):
            # assuming that this is a one-level dictionary
            self._observations = {}
            self._next_obs = {}

            for key, dims in observation_dim.items():
                if isinstance(dims, tuple):
                    dims = tuple([max_replay_buffer_size] + list(dims))
                else:
                    dims = (max_replay_buffer_size, dims)
                self._observations[key] = np.zeros(dims)
                self._next_obs[key] = np.zeros(dims)
        else:
            # else observation_dim is an integer
            self._observations = np.zeros((max_replay_buffer_size, observation_dim))
            self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))

        self._actions = np.zeros((max_replay_buffer_size, action_dim))

        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._data = []
        self._full_data = []
        self._top = 0
        self._size = 0

        self._initial_pairs = []
        self._last_pairs = None

        self.bc_buffer = None
        self.simple_buffer = None
        self.full_buffer = None
        self.used_traj_num = 0

        self.bc_sampling_with_rep = bc_sampling_with_rep
        self.traj_sel_crt = traj_sel_crt
        self.bc_traj_limit = bc_traj_limit
        print("BC TRAJ LIMIT = ", self.bc_traj_limit)

        self.shrink_ratio = 0.0
        self.shrink_update = kwargs.get("shrink_update", 0.2)
        self.only_bc = only_bc
        self.data_aug = data_aug

    def get_size(self):
        return self._size

    def get_new_simple_buffer(self):
        return SimpleReplayBuffer(self._max_sub_buf_size, self._observation_dim, self._action_dim, self._random_seed)

    def get_full_buffer(self):
        try:
            self.full_buffer
        except AttributeError:
            self.full_buffer = None
        if self.full_buffer is None:
            self.full_buffer = self.get_new_simple_buffer()
            for traj in self._data:
                self.full_buffer.add_path(traj)
        return self.full_buffer

    def set_data(self, data, **kwargs):
        """
        Set the buffer with a list-like data.
        """
        self._full_data = data
        self._data = data * self.data_aug
        self._data = self._data[:self._max_replay_buffer_size]
        self._data.sort(key=lambda trajectory: trajectory['ep_rews'])
        filter_percent = kwargs.get('filter_percent', 1)
        self._data = self._data[int(len(self._data) * (1 - filter_percent)):]
        print("USE DATA OF TOP %d%%, DATASET LEN = " % (int(filter_percent * 100)), len(self._data))
        min_rew = self._data[0]["ep_rews"]
        print("MIN REWARD IN DATA = ", min_rew)

        self._top = len(self._data) - 1
        self._size = len(self._data)

        self._initial_pairs = {'observations': [], 'actions': [], 'next_observations': [], 'rewards': []}
        self._last_pairs = {'observations': [], 'actions': [], 'next_observations': [], 'rewards': [], 'gamma_pow': []}

        return_stat = kwargs.get('return_stat', False)
        obs_mean = 0.0
        obs_std = 0.0
        act_mean = 0.0
        act_std = 0.0
        num_pair = 0

        for traj in self._data:
            if return_stat:
                num_pair += len(traj['observations'])
                for obs, act in zip(traj['observations'], traj['actions']):
                    obs_mean += obs
                    act_mean += act
            for key in self._initial_pairs.keys():
                self._initial_pairs[key].append(traj[key][0])
                self._last_pairs[key].append(traj[key][-1])
                self._last_pairs['gamma_pow'].append(self.gamma ** len(traj['observations']))

        if return_stat:
            obs_mean /= num_pair
            act_mean /= num_pair
            for traj in self._data:
                for obs, act in zip(traj['observations'], traj['actions']):
                    obs_std += (obs - obs_mean) ** 2
                    act_std += (act - act_mean) ** 2
            obs_std = np.sqrt(obs_std / num_pair)
            act_std = np.sqrt(act_std / num_pair)

            print("STATISTICS OF DATASET")
            print(obs_mean)
            print(obs_std)
            print(act_mean)
            print(act_std)
            return min_rew, obs_mean, obs_std, act_mean, act_std
        else:
            return min_rew, None, None, None, None

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        :param path: Dict like one output by rlkit.samplers.util.rollout
        """
        # path.keys() = ['actions', 'observations', 'next_observations', 'rewards',
        # 'terminals', 'timeouts', 'ep_rews', "agent_infos", "env_infos",]

        if len(self._data) < self._max_replay_buffer_size:
            self._data.append(path)
            for key in self._initial_pairs.keys():
                self._initial_pairs[key].append(path[key][0])
                self._last_pairs[key].append(path[key][-1])
                self._last_pairs['gamma_pow'].append(self.gamma ** len(path['observations']))
        else:
            print("replace the original buffer")
            self._data[(self._top + 1) % self._max_replay_buffer_size] = path

        self._top += 1
        self._size = min(self._top + 1, self._max_replay_buffer_size)

    def get_pretrain_buffer(self, pretrain_num=-1):
        """
        Return a replay buffer of pretrain data (random trajs).
        :param pretrain_num: the number of trajectories used for pretraining; 
                             if minus, use all trajectories.
        """
        if pretrain_num == 0:
            return self.get_new_simple_buffer()
        if pretrain_num < 0:
            pretrain_buffer = self.get_full_buffer()
            logger.log("Pretrained buf: full buffer, buffer size {}".format(pretrain_buffer.get_size()))
            return pretrain_buffer
        else:
            pretrain_buffer = self.get_new_simple_buffer()
            for i in range(pretrain_num):
                pretrain_buffer.add_path(self._data[i])
            logger.log(
                "Pretrained buf: selected num: {}, buffer size {}".format(pretrain_num, pretrain_buffer.get_size()))
            return pretrain_buffer

    def shrink_buffer(self):
        logger.log("------- SHRINK BUFFER -------")
        if self.shrink_ratio >= 1.0:
            print("======= OUT OF DATASET =======")
            logger.log("======= OUT OF DATESET =======")
            raise ValueError
        if self.simple_buffer is not None:
            del self.simple_buffer
            gc.collect()
        self.simple_buffer = self.get_new_simple_buffer()
        reserved_data = self._full_data[int(len(self._full_data) * self.shrink_ratio):]
        for traj in reserved_data:
            self.simple_buffer.add_path(traj)
        self.shrink_ratio += self.shrink_update

    def has_similar_traj(self, policy_func=None, trajs=None):
        if trajs is None:
            trajs = self._data
        if self.traj_sel_crt == "reward":
            log_ratios = list(reversed(range(len(trajs))))
        else:
            log_ratios = [action_deriviation(traj, policy_func, mode=self.traj_sel_crt) for traj in trajs]
        try:
            max_log_ratio = np.max(log_ratios)
        except ValueError:
            print("======= OUT OF DATASET =======")
            logger.log("======= OUT OF DATESET =======")
            raise ValueError
        return log_ratios, max_log_ratio

    def get_sufficient_buffer(self, policy_func=None, **kwargs):
        """
        Return a replay buffer that sufficient the reward ratio requirement.
        :param policy_func: policy function
        """
        if kwargs.get("shrink_buffer", False):
            return self.simple_buffer, "bc", 0, 0, 0, 0

        log_ratios, max_log_ratio = self.has_similar_traj(policy_func)
        return_trajs = kwargs.get("return_trajs", False)

        bc_traj_num = self.bc_traj_limit
        selected_ind = heapq.nlargest(bc_traj_num, enumerate(log_ratios), key=lambda x: x[1])
        selected_ind, _ = zip(*selected_ind)
        selected_trajs = [self._data[ind] for ind in selected_ind]

        min_reward = np.min([traj["ep_rews"] for traj in selected_trajs])
        self._data = [self._data[ind] for ind in range(len(self._data)) if ind not in selected_ind]

        self.used_traj_num += len(selected_trajs)
        training_mode = "bc"
        if self.bc_buffer is not None:
            del self.bc_buffer
            gc.collect()
        self.bc_buffer = SimpleReplayBuffer(self._max_sub_buf_size, self._observation_dim, self._action_dim,
                                            self._random_seed)

        traj_num = len(selected_trajs)
        traj_steps = 0
        traj_order = []
        traj_rews = []
        for traj in selected_trajs:
            self.bc_buffer.add_path(traj)
            traj_steps += len(traj["terminals"])
            try:
                traj_order.append(traj["order"])
            except KeyError:
                traj_order.append(0)
            traj_rews.append(traj["ep_rews"])
        logger.log(f"Selected num: {traj_num}, Used num: {self.used_traj_num}, Remaining Num: {len(self._data)}, Max log ratio: {max_log_ratio}, ")

        return (self.bc_buffer, selected_trajs) if return_trajs \
            else (self.bc_buffer, training_mode, traj_steps, traj_order, traj_rews, min_reward)

    def get_bc_buffer(self):
        return self.bc_buffer

    def filter_below(self, reward):
        self._data = [traj for traj in self._data if traj["ep_rews"] >= reward]

    def get_obs_bound(self):
        obs_size = self._full_data[0]["observations"].shape[1]
        obs_bound = [[float("inf"), -float("inf")]] * obs_size
        for traj in self._full_data:
            for obs in traj["observations"]:
                for i in range(obs_size):
                    obs_bound[i][0] = min(obs_bound[i][0], obs[i])
                    obs_bound[i][1] = max(obs_bound[i][1], obs[i])
        return obs_bound

    def get_act_bound(self):
        act_size = self._full_data[0]["actions"].shape[1]
        act_bound = [[float("inf"), -float("inf")]] * act_size
        for traj in self._full_data:
            for obs in traj["actions"]:
                for i in range(act_size):
                    act_bound[i][0] = min(act_bound[i][0], obs[i])
                    act_bound[i][1] = max(act_bound[i][1], obs[i])
        return act_bound

    def get_full_data(self):
        return self._full_data

    def calc_discounted_rewards(self):
        for d in range(len(self._data)):
            tmp = []
            gamma_pow = 1.0
            r = 0.0
            for i in range(1, len(self._data[d]["actions"])+1):
                r = gamma_pow * r + self._data[d]["rewards"][-i]
                tmp.append(r)
                gamma_pow *= self.gamma
            self._data[d]["discounted_rewards"] = np.array(list(reversed(tmp)))

    def get_initial_pairs(self):
        """
        Return a list of initial pairs of each traj.
        """

        assert self._size > 0

        return self._initial_pairs

    def get_last_pairs(self):
        """
        Return a list of initial pairs of each traj.
        """

        assert self._size > 0

        return self._last_pairs

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """
        Add a transition tuple.
        """
        pass

    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass


def action_deriviation(traj, policy_func, mode="alpha2"):
    obs = torch.Tensor(traj['observations']).to(ptu.device)
    acts = torch.Tensor(traj['actions']).to(ptu.device)
    log_prob = policy_func.get_log_prob(obs, acts, return_normal_params=True)[0]
    log_prob = log_prob.detach().cpu().squeeze().numpy()
    if np.ndim(log_prob) == 0:
        log_prob = np.reshape(log_prob, 1)

    res = 1.0

    if mode == "is":
        res = np.min(log_prob)
    elif mode == "mean":
        res = np.mean(log_prob)
    elif mode == "median":
        res = np.median(log_prob)
    elif mode.startswith("alpha"):
        t = int(mode[5:])
        np.sort(log_prob)
        t_list = [0, 0.6827, 0.9545, 0.9974]
        t_id = int(t_list[t] * len(log_prob))
        res = np.partition(log_prob, -t_id)[-t_id]
    elif mode == "prod":
        res = np.sum(log_prob)

    return res

import yaml
import argparse
import joblib
import numpy as np
import os
import sys
import inspect
import pickle
from gym.spaces import Dict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

from rlkit.envs import get_env
from rlkit.launchers.launcher_util import setup_logger, set_seed
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.sac.modified_sac_ablation import SoftActorCritic
from rlkit.torch.phase_offline.phase_offline import PhaseOffline
from rlkit.envs.wrappers import ScaledEnv
from rlkit.launchers import config
from rlkit.data_management.episodic_replay_buffer import EpisodicReplayBuffer

random_benchmark_rews = {'hopper': 7, }

def experiment(variant):
    with open('demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)
    demos_path = listings[variant['dataset_name']]['file_paths'][0]
    print("demos_path", demos_path)
    with open(demos_path, 'rb') as f:
        traj_list = pickle.load(f)

    env_specs = variant['env_specs']
    env_name = env_specs['env_name']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])
    training_env = get_env(env_specs)
    training_env.seed(env_specs['training_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))

    max_replay_buffer_size = len(traj_list) * 10
    max_sub_buf_size = variant['sub_buf_size']
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = EpisodicReplayBuffer(max_replay_buffer_size, max_sub_buf_size, observation_dim, action_dim,
                                         env_specs['eval_env_seed'], variant['sac_params']['discount'],
                                         variant['use_epsilon_decay'], variant['epsilon_min_or_init'],
                                         variant['epsilon_decay'], variant['bc_sampling_with_rep'])
    replay_buffer.set_data(traj_list)
    
    if variant['scale_env_with_demo_stats']:
        env = ScaledEnv(
            env,
            obs_mean=None,
            obs_std=None,
            acts_mean=None,
            acts_std=None,
        )
        training_env = ScaledEnv(
            training_env,
            obs_mean=None,
            obs_std=None,
            acts_mean=None,
            acts_std=None,
        )

    obs_space = env.observation_space
    act_space = env.action_space
    assert not isinstance(obs_space, Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1
    
    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']
    qfs = []
    for _ in range(10):
        qfs.append(FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1
        ))
    vf = FlattenMlp(
        hidden_sizes=num_hidden * [net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    
    # set up the algorithm
    trainer = SoftActorCritic(
        policy=policy,
        qfs=qfs,
        vf=vf,
        **variant['sac_params']
    )
    
    algorithm = PhaseOffline(
        env=env,
        training_env=training_env,
        exploration_policy=policy,

        policy_trainer=trainer,
        replay_buffer=replay_buffer,
        random_reward=random_benchmark_rews[env_name],
        **variant['offline_params']
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    algorithm.train()

    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs['exp_id']
    exp_prefix = exp_specs['exp_name']
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs)

    experiment(exp_specs)

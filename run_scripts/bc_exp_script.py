import yaml
import argparse
import os,sys,inspect
import pickle

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
print(sys.path)

from gym.spaces import Dict
from rlkit.envs import get_env

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed

from rlkit.envs.wrappers import ScaledEnv
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.bc.bc import BC
from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer

def experiment(variant):
    with open('demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read(), Loader=yaml.FullLoader)
    expert_demos_path = listings[variant['expert_name']]['file_paths'][variant['expert_idx']]
    with open(expert_demos_path, "rb") as f:
        traj_list = pickle.load(f)
    filter_percent = variant.get("filter_percent", 1.00)
    traj_list = traj_list[int(len(traj_list) * (1 - filter_percent)):]
    print("USE DATA OF TOP %d%%, DATASET LEN = " % (int(filter_percent * 100)), len(traj_list))

    env_specs = variant['env_specs']
    env = get_env(env_specs)
    env.seed(env_specs['eval_env_seed'])
    training_env = get_env(env_specs)
    training_env.seed(env_specs['training_env_seed'])

    print('\n\nEnv: {}'.format(env_specs['env_name']))
    print('kwargs: {}'.format(env_specs['env_kwargs']))
    print('Obs Space: {}'.format(env.observation_space))
    print('Act Space: {}\n\n'.format(env.action_space))

    max_replay_buffer_size = len(traj_list) * 10
    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    expert_replay_buffer = SimpleReplayBuffer(max_replay_buffer_size, observation_dim, action_dim, seed)
    for traj in traj_list:
        expert_replay_buffer.add_path(traj)

    if variant['scale_env_with_demo_stats']:
        print("Use scaled environment")
        obs_mean, obs_std, acts_mean, acts_std = expert_replay_buffer.get_stats()
        expert_replay_buffer.scale_data(obs_mean, obs_std, None, None)
        env = ScaledEnv(
            env,
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=None,
            acts_std=None,
        )
        training_env = ScaledEnv(
            training_env,
            obs_mean=obs_mean,
            obs_std=obs_std,
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
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )

    algorithm = BC(
        env=env,
        training_env=training_env,
        exploration_policy=policy,

        expert_replay_buffer=expert_replay_buffer,
        **variant['bc_params']
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
        exp_specs = yaml.load(spec_string, Loader=yaml.FullLoader)

    # make all seeds the same.
    exp_specs['env_specs']['eval_env_seed'] = exp_specs['env_specs']['training_env_seed'] = exp_specs['seed']

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs['exp_id']
    exp_prefix = "%s-tp{%s}-ex{%s}" % (
        exp_specs['exp_name'],
        exp_specs['filter_percent'],
        exp_specs.get('extra_info', "")
    )
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)

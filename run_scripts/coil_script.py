import yaml
import argparse
import joblib
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
from rlkit.torch.coil.coil import COIL
from rlkit.torch.sac.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.phase_offline.phase_offline_coil import PhaseOffline
from rlkit.envs.wrappers import ScaledEnv
from rlkit.data_management.episodic_replay_buffer_coil import EpisodicReplayBuffer


def experiment(variant, **kwargs):
    with open('demos_listing.yaml', 'r') as f:
        listings = yaml.load(f.read())
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

    variant.setdefault('use_epsilon_decay', False)
    variant.setdefault('epsilon_decay', 1)
    variant.setdefault('bc_sampling_with_rep', True)
    variant.setdefault('only_bc', False)
    variant.setdefault('data_aug', 1)
    replay_buffer = EpisodicReplayBuffer(
        max_replay_buffer_size, max_sub_buf_size, observation_dim, action_dim,
        random_seed=env_specs['eval_env_seed'], 
        bc_sampling_with_rep=variant['bc_sampling_with_rep'], 
        traj_sel_crt=variant['traj_sel_crt'],
        bc_traj_limit=variant['bc_traj_limit'], 
        only_bc=variant['only_bc'], 
        data_aug=variant['data_aug']
    )
    filter_percent = variant.get('filter_percent', 1)
    return_stat = variant['scale_env_with_demo_stats']
    min_rew, obs_mean, obs_std, acts_mean, acts_std = replay_buffer.set_data(traj_list, filter_percent=filter_percent,
                                                                             return_stat=return_stat)
    replay_buffer.calc_discounted_rewards()

    if variant['scale_env_with_demo_stats']:
        print("SCALE ENV")
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
    gaussian_std = variant.get('gaussian_std', None)
    dropout = variant.get('dropout', None)

    # build the policy models
    net_size = variant['policy_net_size']
    num_hidden = variant['policy_num_hidden_layers']
    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
        std=gaussian_std,
        dropout=dropout
    )

    # set up the algorithm
    trainer = COIL(
        policy=policy,
        **variant['coil_params']
    )

    algorithm = PhaseOffline(
        env=env,
        training_env=training_env,
        exploration_policy=policy,

        policy_trainer=trainer,
        replay_buffer=replay_buffer,
        min_reward=min_rew,
        **variant['offline_params']
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)

    pre_epoch = 0
    algorithm.train(start_epoch=pre_epoch)
    return 1


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', help='experiment specification file')
    parser.add_argument('-g', '--gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--snapshot', help='load from snapshot')
    args = parser.parse_args()
    with open(args.experiment, 'r') as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.load(spec_string)

    snapshot = None
    extra_data = None
    snapshot_path = args.snapshot
    if snapshot_path is not None:
        extra_data_path = snapshot_path.replace("itr", "extra_data")
        snapshot = joblib.load(snapshot_path)
        extra_data = joblib.load(extra_data_path)

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    exp_specs['env_specs']['eval_env_seed'] = exp_specs['env_specs']['training_env_seed'] = exp_specs['seed']

    if exp_specs['num_gpu_per_worker'] > 0:
        print('\n\nUSING GPU\n\n')
        ptu.set_gpu_mode(True, args.gpu)
    exp_id = exp_specs['exp_id']
    exp_prefix = "%s-ft{%s}-nb{%s}-bc{%s}-ex{%s}" % (
        exp_specs['exp_name'],
        exp_specs['offline_params']['filter_tau'],
        exp_specs['offline_params']['num_bc_update_loops_per_train_call'],
        exp_specs['bc_traj_limit'],
        exp_specs.get('extra_info', "")
    )
    seed = exp_specs['seed']
    set_seed(seed)
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, log_tboard=False)

    experiment(exp_specs, snapshot=snapshot, extra_data=extra_data)

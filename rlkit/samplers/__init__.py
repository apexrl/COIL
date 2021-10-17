import numpy as np
from rlkit.data_management.path_builder import PathBuilder


def rollout(
        env,
        policy,
        max_path_length,
        no_terminal=False,
        render=False,
        pred_obs=False,
        render_kwargs={},
        render_mode='rgb_array',
):
    path_builder = PathBuilder()
    observation = env.reset()
    pred_obs_prime = None

    images = []
    image = None
    for _ in range(max_path_length):
        if pred_obs:
            pred_obs_prime, action, agent_info = policy.get_action(observation, return_predicting_obs=pred_obs)
        else:
            action, agent_info = policy.get_action(observation)
        if render:
            if render_mode == 'rgb_array':
                image = env.render(mode=render_mode, **render_kwargs)
                images.append(image)
            else:
                env.render(**render_kwargs)

        next_ob, reward, terminal, env_info = env.step(action)
        if no_terminal: terminal = False

        # print(pred_obs_prime, next_ob)

        if terminal:
            path_builder.add_all(
                observations=observation,
                actions=action,
                rewards=np.array([reward]),
                predicted_observations=pred_obs_prime,
                next_observations=next_ob,
                terminals=np.array([terminal]),
                absorbing=np.array([0., 1.]),
                agent_info=agent_info,
                env_info=env_info,
                image=image,
            )
            path_builder.add_all(
                observations=observation,
                actions=action,
                rewards=np.array([reward]),
                predicted_observations=pred_obs_prime,
                next_observations=next_ob,
                terminals=np.array([terminal]),
                absorbing=np.array([1., 1.]),
                agent_info=agent_info,
                env_info=env_info,
                image=image,
            )

        else:
            path_builder.add_all(
                observations=observation,
                actions=action,
                rewards=np.array([reward]),
                predicted_observations=pred_obs_prime,
                next_observations=next_ob,
                terminals=np.array([terminal]),
                absorbing=np.array([0., 0.]),
                agent_info=agent_info,
                env_info=env_info,
                image=image,
            )

        observation = next_ob
        if terminal: break
    return path_builder


class PathSampler():
    def __init__(
            self,
            env,
            policy,
            num_steps,
            max_path_length,
            no_terminal=False,
            render=False,
            render_kwargs={},
            render_mode='rgb_array'
    ):
        """
        When obtain_samples is called, the path sampler will generates the
        minimum number of rollouts such that at least num_steps timesteps
        have been sampled
        """
        self.env = env
        self.policy = policy
        self.num_steps = num_steps
        self.max_path_length = max_path_length
        self.no_terminal = no_terminal
        self.render = render
        self.render_kwargs = render_kwargs
        self.render_mode = render_mode

    def obtain_samples(self, num_steps=None, pred_obs=False):
        paths = []
        total_steps = 0
        if num_steps is None:
            num_steps = self.num_steps
        while total_steps < num_steps:
            new_path = rollout(
                self.env,
                self.policy,
                self.max_path_length,
                no_terminal=self.no_terminal,
                pred_obs=pred_obs,
                render=self.render,
                render_kwargs=self.render_kwargs,
                render_mode=self.render_mode
            )
            paths.append(new_path)
            total_steps += len(new_path['rewards'])
        return paths

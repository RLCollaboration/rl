import gym
from collections import namedtuple

env_transition = namedtuple('Transition', ['state', 'action', 'next_state', 'reward', 'is_terminal'], verbose=False)


class EnvWrapper(object):
    """ Simple Wrapper for OpenAI gym environments to facilitate interactions with tensorflow.  The
        class accepts the name of an OpenAI gym environment, creates the environment, and then
        supports interactions against the environment.

        Args:
            name (str): The name of an OpenAI gym environment. (For example, 'CartPole-v0')
            obs_hook (callable): A callback function that transforms observations from the environment into a more
                                 convenient data structure.
    """

    def __init__(self, name, obs_hook=lambda x: x, log_hook=None, render=False):

        self._env = gym.make(name)
        self._current_obs = None

        self.obs_hook = obs_hook
        self.log_hook = log_hook
        self.render = render

        # Reset environment and choose new initial state
        self.reset()

    def step(self, action):
        next_obs, reward, is_terminal, info = self._env.step(action)

        # Render the OpenAI gym environment
        if self.render:
            self._env.render()

        # Output diagnostic information (if log_hook supplied)
        if self.log_hook:
            self.log_hook(info)

        # If this is a terminal state then reset the environment
        if is_terminal:
            self.reset()

        # Current/Next states are the transformed observations
        current_state = self.current_state
        next_state = self.obs_hook(next_obs)

        trans = env_transition(state=current_state,
                               action=action,
                               reward=reward,
                               next_state=next_state,
                               is_terminal=is_terminal)

        self._current_obs = next_obs

        return trans

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def current_state(self):
        """ Returns the current state from the environment.

        The current state is the last environment observation optionally processed by the obs_hook.
        """
        return self.obs_hook(self._current_obs)

    def reset(self):
        self._current_obs = self._env.reset()

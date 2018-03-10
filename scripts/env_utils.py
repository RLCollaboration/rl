import gym


class EnvWrapper(object):
    """ Simple Wrapper for OpenAI gym environments to facilitate interactions with tensorflow.  The
        class accepts the name of an OpenAI gym environment, creates the environment, and then
        supports interactions against the environment.

        Args:
            name (str): The name of an OpenAI gym environment. (For example, 'CartPole-v0')
            obs_hook (callable): A callback function that transforms observations from the environment into a more
                                 convenient data structure.
    """
    def __init__(self, name, obs_shape=None, obs_hook=None):

        self.env = gym.make(name)
        self.env.reset()

        self.obs_size = obs_shape
        self.obs_hook = obs_hook or (lambda x: x)

    def step(self, action):
        s, r, done, info = self.env.step(action)

        # TODO: Log info?

        return self.obs_hook(s), r, done

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        self.env.reset()



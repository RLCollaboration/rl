import unittest
import gym

from env_utils import EnvWrapper

TEST_ENV_NAME = 'CartPole-v0'


class TestEnvWrapper(unittest.TestCase):


    def test_env_create(self):
        # Test empty environment name raises exception
        with self.assertRaises(Exception):
            env = EnvWrapper()

        # Test unsupported environment name raises exception
        with self.assertRaises(Exception):
            env = EnvWrapper(name='bad env')

        # Test supported environment name
        try:
            env = EnvWrapper(name=TEST_ENV_NAME)
        except Exception as e:
            self.fail(e.message)

        # Test observation hook argument
        try:
            env = EnvWrapper(name=TEST_ENV_NAME, obs_hook=lambda x: 1)
        except Exception as e:
            self.fail(e.message)

    def test_step(self):

        # Test step with observation hook
        env = EnvWrapper(TEST_ENV_NAME, obs_hook=lambda x: 1)
        s, r, done = env.step(env.action_space.sample())
        self.assertEqual(s, 1)




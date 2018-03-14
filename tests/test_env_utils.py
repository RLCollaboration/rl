import unittest
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

        # Random current state should be a 4 tuples (if no obs_hook)
        self.assertEqual(len(env.current_state), 4)

        # Test observation hook argument
        try:
            env = EnvWrapper(name=TEST_ENV_NAME, obs_hook=lambda x: 1)
        except Exception as e:
            self.fail(e.message)

        # Current state should 1 with this obs_hook
        self.assertEqual(env.current_state, 1)

    def test_step(self):

        # Test step with observation hook
        env = EnvWrapper(TEST_ENV_NAME, obs_hook=lambda x: 1)
        trans = env.step(env.action_space.sample())
        self.assertEqual(trans.next_state, 1)

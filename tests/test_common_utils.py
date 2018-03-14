import unittest

from mock import patch
from common_utils import eps_greedy

import numpy as np


class FakeDNN(object):
    def __call__(self, *args, **kwargs):
        return [np.array([0.1, 0.2, .03])]


class FakeEnv(object):
    def current_state(self):
        return [1.0]

    @property
    def action_space(self):
        class FakeActionSpace(object):
            def sample(self):
                return 17

        return FakeActionSpace()


class MyTestCase(unittest.TestCase):
    # Test greedy action (random value < epsilon)
    def test_eps_greedy_greedy_action(self):
        with patch('common_utils.random', return_value=0.01) as mock_random:
            Q = FakeDNN()
            env = FakeEnv()
            self.assertEqual(eps_greedy(env=env, epsilon=0.9, Q=Q), 1)

    # Test random action (random value >= epsilon)
    def test_eps_greedy_random_action(self):
        with patch('common_utils.random', return_value=0.99) as mock_random:
            Q = FakeDNN()
            env = FakeEnv()
            self.assertEqual(eps_greedy(env=env, epsilon=0.9, Q=Q), 17)


if __name__ == '__main__':
    unittest.main()

import unittest

from mock import patch
from common_utils import eps_greedy

import numpy as np
import tensorflow as tf

from datatypes import DeepNN
from env_utils import EnvWrapper


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
    def test_eps_greedy_greedy_action1(self):
        with patch('common_utils.random', return_value=0.01) as mock_random:
            Q = FakeDNN()
            env = FakeEnv()
            self.assertEqual(eps_greedy(env=env, epsilon=0.9, Q=Q), 1)

    # Test greedy action (random value < epsilon)
    def test_eps_greedy_greedy_action2(self):
        with patch('common_utils.random', return_value=0.9) as mock_random:
            Q = FakeDNN()
            env = FakeEnv()
            self.assertEqual(eps_greedy(env=env, epsilon=0.9, Q=Q), 1)

    # Test random action (random value > epsilon)
    def test_eps_greedy_random_action(self):
        with patch('common_utils.random', return_value=0.91) as mock_random:
            Q = FakeDNN()
            env = FakeEnv()
            self.assertEqual(eps_greedy(env=env, epsilon=0.9, Q=Q), 17)

    def test_with_tensorflow_and_real_env(self):

        try:

            env = EnvWrapper(name='CartPole-v0')

            g = tf.Graph()
            with g.as_default():
                s = tf.placeholder(shape=(None, len(env.current_state)), dtype=tf.float32)
                Q = DeepNN(name='Q_target',
                           inputs=s,
                           hidden_layer_sizes=[10, 5],
                           n_actions=env.action_space.n)
                init = tf.global_variables_initializer()

            with tf.Session(graph=g) as sess:
                sess.run(init)

                # Force greedy policy to guarantee Q is evaluated
                eps_greedy(Q=Q, env=env, epsilon=1.0)

        except Exception as e:
            self.fail(e.message)


if __name__ == '__main__':
    unittest.main()

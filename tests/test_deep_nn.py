import unittest
import tensorflow as tf
import numpy as np

from collections import namedtuple

from datatypes import DeepNN
from random import random

test_scenario = namedtuple('scenario', ['name', 'inputs', 'layer_sizes', 'n_actions', 'trainable'])


class TestDeepNN(unittest.TestCase):
    def test_create_dnn(self):
        # Check single input, single hidden layer unit, single output
        with tf.Graph().as_default():
            scenario = test_scenario(name='NN1',
                                     inputs=tf.placeholder(name='s', dtype=tf.float32, shape=(None, 1)),
                                     layer_sizes=[1],
                                     n_actions=1,
                                     trainable=True)

            self._execute_test_scenario(scenario)

        # Check multiple inputs, deep/wide network, multiple outputs
        with tf.Graph().as_default():
            scenario = test_scenario(name='NN2',
                                     inputs=tf.placeholder(name='s', dtype=tf.float32, shape=(None, 1000)),
                                     layer_sizes=[500, 250, 175, 100, 50],
                                     n_actions=25,
                                     trainable=True)

            self._execute_test_scenario(scenario)

    def _execute_test_scenario(self, scenario):
        dnn = DeepNN(name=scenario.name,
                     inputs=scenario.inputs,
                     hidden_layer_sizes=scenario.layer_sizes,
                     n_actions=scenario.n_actions,
                     trainable=scenario.trainable)

        # Verify all class properties set correctly
        self.assertEqual(dnn.name, scenario.name)
        self.assertEqual(dnn.inputs, scenario.inputs)
        self.assertEqual(dnn.hidden_layer_sizes, scenario.layer_sizes)
        self.assertEqual(dnn.n_actions, scenario.n_actions)
        self.assertEqual(dnn.trainable, scenario.trainable)

        # Check that variables property is set
        self.assertIsNotNone(dnn.variables)

        # Check correct number of parameters (weights and biases)
        units_per_layer = [scenario.inputs.get_shape()[1]] + scenario.layer_sizes + [scenario.n_actions]
        weights = np.sum([m * n for m, n in zip(units_per_layer, units_per_layer[1:])])
        biases = np.sum(units_per_layer[1:])

        expected = weights + biases
        actual = np.sum([np.product(v.get_shape()) for v in dnn.variables])

        self.assertEqual(actual, expected, msg='Excepted {} variables but found {}'.format(expected, actual))

        # Check that action values property is set (output layer units)
        self.assertIsNotNone(dnn.action_values)

        # Check for the correct number of action values
        expected = scenario.n_actions
        actual = dnn.action_values.get_shape()[1]

        self.assertEqual(actual, expected, msg='Excepted {} action values but found {}'.format(expected, actual))

        # Check forward pass
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Single state input
            state = [[random() for _ in xrange(dnn.inputs.shape[1])]]
            action_values = dnn(state)

            # Verify single state input produces single set of action value outputs
            self.assertEqual(len(action_values), 1)

            # Verify that the number of action values in array returned matches the expected shape
            self.assertEqual(action_values[0].shape[1], dnn.action_values.get_shape()[1])

            # Multiple state inputs
            states = [[random() for _ in xrange(dnn.inputs.shape[1])]]
            action_values = dnn(state)

            # Verify multiple states produces list of action value arrays with one array per input state
            self.assertEqual(len(action_values), len(states))

            # Verify that each array of action values returned has the expected number of action values
            for a in action_values:
                self.assertEqual(a.shape[1], dnn.action_values.get_shape()[1])

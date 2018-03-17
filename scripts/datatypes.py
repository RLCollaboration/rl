import tensorflow as tf
import numpy as np

from random import random, sample


class DeepNN(object):
    def __init__(self, name, inputs, hidden_layer_sizes, n_actions, trainable=True):
        self.name = name
        self.inputs = inputs
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_actions = n_actions
        self.trainable = trainable

        self.variables = []
        self.action_values = None

        # Create the DNN
        with tf.variable_scope(name):
            previous = self.inputs

            # Create the Hidden Layers
            for i, layer_size in enumerate(hidden_layer_sizes):
                previous, w, b = self._create_layer("Hidden_{}".format(i),
                                                    previous, layer_size,
                                                    activation=tf.nn.relu)
                self.variables += [w, b]

            # Create the Final Layer
            self.action_values, w, b = self._create_layer("Final", previous, n_actions, activation=None)
            self.variables += [w, b]

    def _create_layer(self, scope_name, inputs, layer_size, activation=None):
        with tf.variable_scope(scope_name):
            input_size = inputs.get_shape()[1]

            w = tf.get_variable("weights",
                                shape=(input_size, layer_size),
                                trainable=self.trainable,
                                dtype=tf.float32)

            b = tf.get_variable("bias",
                                shape=(layer_size,),
                                initializer=tf.zeros_initializer(),
                                trainable=self.trainable,
                                dtype=tf.float32)

            output = tf.matmul(inputs, w) + b

            if activation:
                output = activation(output)

            return output, w, b

    def __call__(self, *args, **kwargs):
        states = list(args[0]) if isinstance(args[0], list) else args[0]
        sess = kwargs['session'] \
            if 'session' in kwargs and kwargs['session'] is not None \
            else tf.get_default_session()
        return sess.run([self.action_values], feed_dict={self.inputs: states})


class ReplayBuffer(object):
    def __init__(self, max_size=100):
        self.max_size = max_size

        self._elements = []
        self._last_ndx = -1

    def __len__(self):
        return len(self._elements)

    def full(self):
        return len(self) == self.max_size

    def append(self, obj):
        self._last_ndx = (self._last_ndx + 1) % self.max_size
        if len(self) < self.max_size:
            self._elements.append(obj)
        else:
            self._elements[self._last_ndx] = obj

    def count(self, obj):
        return self._elements.count(obj)

    def sample(self, size):
        return sample(self._elements, size)

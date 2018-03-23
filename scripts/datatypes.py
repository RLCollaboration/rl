from collections import namedtuple

import numpy as np
import tensorflow as tf


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


Batch = namedtuple('Batch', ['states', 'actions', 'rewards', 'next_states', 'is_terminal_indicators'],
                   verbose=False)


class ReplayBuffer(object):
  def __init__(self, state_dim, max_size=100):
    self.max_size = max_size

    self._states = np.zeros((self.max_size, state_dim))
    self._actions = np.zeros(self.max_size)
    self._rewards = np.zeros(self.max_size)
    self._next_states = np.zeros((self.max_size, state_dim))
    self._is_terminal_states = np.zeros(self.max_size)

    self._last_ndx = -1
    self._size = 0

  def __len__(self):
    return self._size

  def full(self):
    return len(self) == self.max_size

  def append(self, trans):
    self._last_ndx = (self._last_ndx + 1) % self.max_size

    if len(self) < self.max_size:
      self._size += 1

    self._states[self._last_ndx] = trans.state
    self._actions[self._last_ndx] = trans.action
    self._rewards[self._last_ndx] = trans.reward
    self._next_states[self._last_ndx] = trans.next_state
    self._is_terminal_states[self._last_ndx] = 1.0 if trans.is_terminal else 0.0

  def __getitem__(self, indexes):
    return Batch(states=self._states[indexes],
                 actions=self._actions[indexes],
                 rewards=self._rewards[indexes],
                 next_states=self._next_states[indexes],
                 is_terminal_indicators=self._is_terminal_states[indexes])

  def sample(self, sample_size):
    indexes = np.random.choice(np.arange(self._size), size=sample_size, replace=False)
    return self[indexes]

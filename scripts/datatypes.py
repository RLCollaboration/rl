import numpy as np
import tensorflow as tf

from collections import namedtuple


class ConvLayer:
  def __init__(self, size, type='tf.layers.Conv2D', strides=(1, 1), padding='same', activation=tf.nn.relu):
    self.size = size
    self.type = type
    self.strides = strides
    self.padding = padding
    self.activation = activation


# def create_cnn(name, inputs, input_size, conv_filters, max_pool_layer_strides, n_actions, trainable=True):

def create_cnn(name, inputs, input_sizes, layers, n_actions, trainable=True):
  variables = []

  output = inputs
  output_sizes = input_sizes

  last_conv_size = 0

  for layer in layers:

    if layer.type == 'tf.layers.Conv2D':

      conv_layer = tf.layers.Conv2D(
        filters=layer.size,
        kernel_size=[5, 5],
        padding=layer.padding,
        activation=layer.activation,
        trainable=trainable)

      output = conv_layer(output)
      variables += conv_layer.variables
      last_conv_size = layer.size

    elif layer.type == 'tf.layers.max_pooling2d':

      output = tf.layers.max_pooling2d(inputs=output, pool_size=layer.size, strides=layer.strides)
      output_sizes = [int(output_sizes[0] / layer.strides[0]), int(output_sizes[1] / layer.strides[1])]

  output_size = output_sizes[0] * output_sizes[1] * last_conv_size

  flattened_output = tf.reshape(output, [-1, output_size])

  dense = tf.layers.Dense(units=1024, activation=tf.nn.relu)
  variables += dense.variables

  output = dense(flattened_output)

  # CNN output size: (160 - 5 + 4) + 1
  # CNN output size: (210 - 5 + 4) + 1

  # dropout = tf.layers.dropout(
  #   inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.Dense(units=n_actions)

  return logits(output), variables


def create_deep_conv_net(name, inputs, hidden_layer_sizes, n_actions, batch_size, trainable=True):
  variables = []

  with tf.variable_scope(name):
    previous = inputs

    # Create the Hidden Layers
    for i, layer_size in enumerate(hidden_layer_sizes):
      previous, v = create_conv_layer("Conv_{}".format(i),
                                      previous, n_filters=layer_size,
                                      activation=tf.nn.relu,
                                      trainable=trainable)
      variables += v

    output_shape = previous.get_shape().as_list()[1:]
    last_dim_size = 1
    for v in output_shape:
      last_dim_size *= v

    # flattened_shape = tf.stack(values=[batch_size, last_dim_size])

    # Dense Layers
    flattened_layer = tf.reshape(previous, [batch_size, last_dim_size])
    dense, w, b = create_dense_layer("Final", inputs=flattened_layer, layer_size=n_actions, trainable=trainable)
    variables += [w, b]

    # dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=trainable)
    return dense, variables


def create_conv_layer(scope_name, inputs, n_filters, activation=None, trainable=True):
  with tf.variable_scope(scope_name):
    conv_layer = tf.layers.Conv2D(
      filters=n_filters,
      kernel_size=[5, 5],
      padding="same",
      activation=activation,
      trainable=trainable)

    outputs = conv_layer(inputs)

    print(outputs.get_shape().as_list())
    return outputs, conv_layer.variables


def create_dense_layer(scope_name, inputs, layer_size, trainable=True, activation=None):
  with tf.variable_scope(scope_name):
    input_size = inputs.get_shape().as_list()
    print(input_size)
    w = tf.get_variable("weights",
                        shape=(input_size[1], layer_size),
                        trainable=trainable,
                        dtype=tf.float32)

    b = tf.get_variable("bias",
                        shape=(layer_size,),
                        initializer=tf.zeros_initializer(),
                        trainable=trainable,
                        dtype=tf.float32)

    output = tf.matmul(inputs, w) + b

    if activation:
      output = activation(output)

    return output, w, b


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
  def __init__(self, env, max_size=100):
    self.max_size = max_size

    self._states = np.zeros(shape=(max_size,) + env.current_state.shape)
    self._actions = np.zeros(shape=(max_size,))
    self._rewards = np.zeros(shape=(max_size,))
    self._next_states = np.zeros(shape=(max_size,) + env.current_state.shape)
    self._is_terminal_states = np.zeros(shape=(max_size,))

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

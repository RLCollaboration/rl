#! /usr/bin/env python
"""A Tensorflow version of DQN (see https://deepmind.com/research/dqn/)
"""
import tensorflow as tf
import numpy as np
import time

from datetime import datetime

from matplotlib.mlab import cohere_pairs

from common_utils import eps_greedy
from env_utils import EnvWrapper
from datatypes import ReplayBuffer, DeepNN

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/train', 'Directory where to write event logs and checkpoints.')

# Flags for logging
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'How often to log results to the console.')

# Flags for creation of computational graph
tf.app.flags.DEFINE_list('dqn_layer_sizes', [500, 100, 10], 'Layer sizes for the dqn.')

# Flags for environment handling
tf.app.flags.DEFINE_string('environment', 'CartPole-v0', 'The name of the openai gym environment to use')
tf.app.flags.DEFINE_boolean('render', True, 'Whether to render a display of the environment state.')

# Flags for termination criteria
tf.app.flags.DEFINE_integer('steps', 1000000, 'Max number of environment steps (potentially across multiple episodes).')

# Flags for algorithm parameters
tf.app.flags.DEFINE_float('learning_rate', 0.05, 'The learning rate (alpha) to be used.')
tf.app.flags.DEFINE_float('gamma', 0.9, 'The discount factor (gamma) to be used.')
tf.app.flags.DEFINE_float('epsilon', 1.0, 'The initial exploration factor (epsilon) to be used.')

tf.app.flags.DEFINE_integer('buffer_size', 1000, 'Maximum replay buffer size.')
tf.app.flags.DEFINE_integer('mini_batch_size', 64, 'Size of minibatches sampled from replay buffer.')

tf.app.flags.DEFINE_integer('train_q_freq', 50, 'The number of steps before updating the action Q network.')
tf.app.flags.DEFINE_integer('target_q_update_freq', 100,
                            'The number of steps before updating target Q network from action Q network.')

# Shared variables
replay_buffer = ReplayBuffer(FLAGS.buffer_size)
alpha = FLAGS.alpha
gamma = FLAGS.gamma
epsilon = FLAGS.epsilon


class Model(object):
  def __init__(self, gamma, learning_rate, epsilon, env):
    self.gamma = gamma
    self.learning_rate = learning_rate
    self.epsilon = epsilon

    self.env = env
    self.r = None
    self.s = None
    self.s_next = None
    self.a = None
    self.is_terminal = None

    self.update_q_target = None

    self.q_action = None
    self.q_target = None
    self.greedy_action = None
    self.action = None

    self.global_step = None
    self.action_step = None
    self.inc_action_step = None

  def create(self):
    self.global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar("global_step", self.global_step)
    self.action_step = tf.get_variable("action_step", [], dtype=tf.int64)
    tf.summary.scalar("action_step", self.action_step)


    self.r = tf.placeholder(name="reward", dtype=tf.float32, )
    self.s = tf.placeholder(name='state', dtype=tf.float32, shape=(None, len(self.env.current_state)))
    self.s_next = tf.placeholder(name='state_next', dtype=tf.float32, shape=(None, len(self.env.current_state)))
    self.a = tf.placeholder(name='action', dtype=tf.int64)
    self.is_terminal = tf.placeholder(name='is_terminal', dtype=tf.float32)

    self.q_action = DeepNN(name='Q_action',
                           inputs=self.s,
                           hidden_layer_sizes=FLAGS.dqn_layer_sizes,
                           n_actions=self.env.action_space.n)

    self.q_target = DeepNN(name='Q_target',
                           inputs=self.s_next,
                           hidden_layer_sizes=FLAGS.dqn_layer_sizes,
                           n_actions=self.env.action_space.n,
                           trainable=False)

    # Determine next action using an epsilon greedy policy based on Q(S,A)
    self.greedy_action = tf.argmax(self.q_action.action_values, axis=-1)

    self.inc_action_step = tf.assign_add(self.action_step, 1)
    with tf.control_dependencies([self.inc_action_step]):
      self.action = tf.cond(tf.less(tf.random_uniform(shape=[]), self.epsilon), lambda: self.greedy_action,
                            lambda: tf.random_uniform(maxval=self.env.action_space.n, dtype=tf.int64))

    # Calculate Loss
    self.td_target = self.r + self.gamma * tf.maximum(self.q_target.action_values, axis=-1) * (1 - self.is_terminal)
    self.predicted_q = self.q_action.action_values[:, self.a]
    self.loss = tf.losses.mean_squared_error(labels=self.td_target, predictions=self.predicted_q)
    tf.summary.scalar("loss", self.loss)

    self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.train = self.optimizer.minimize(self.loss)

    # Create a copy operation for Q_action to Q_target
    copy_vars_ops = []

    for src, dest in zip(self.q_action.variables, self.q_target.variables):
      copy_vars_ops.append(tf.assign(dest, src))

    self.update_q_target = tf.group(copy_vars_ops, name='update_q_target')


def train(env):
  """Train for a number of steps."""

  model = Model(FLAGS.gamma, FLAGS.learning_rate, FLAGS.epsilon, env)
  model.create()
  # Utility class that creates the tf session for you plus add on functionality
  # -- Different flavors of this exist
  with tf.train.SingularMonitoredSession(
      # save/load model state
      checkpoint_dir=FLAGS.train_dir,
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.steps),
             tf.train.NanTensorHook(model.loss), tf.train.SummarySaverHook(
          save_steps=100,
          save_secs=None,
          output_dir=FLAGS.train_dir,
          summary_writer=None,
          scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()),
          summary_op=None
        )],

      # Can be configured for multi-machine training (check out docs for this class)
      config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement)) as mon_sess:

    while not mon_sess.should_stop():
      # Execute action against environment and observe transition
      action, action_step = mon_sess.raw_session().run([model.action, model.action_step])
      transition = env.step(action)

      replay_buffer.append(transition)

      if replay_buffer.full():
        if action_step % FLAGS.target_q_update_freq == 0:
          mon_sess.raw_session.run(model.update_q_target)

        if action_step % FLAGS.train_q_freq == 0:
          # TODO: Sample minibatch from the replay buffer
          mini_batch = replay_buffer.sample(FLAGS.mini_batch_size)
          # TODO: Update input Dic
          _,loss, global_step = mon_sess.run([model.train_op, model.loss, model.global_step],
                                             {model.r: mini_batch})


def main(argv=None):  # pylint: disable=unused-argument
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  env = EnvWrapper(FLAGS.environment, render=FLAGS.render)
  train(env)


if __name__ == '__main__':
  tf.app.run()

#! /usr/bin/env python
"""A Tensorflow version of DQN (see https://deepmind.com/research/dqn/)
"""
import tensorflow as tf

from datatypes import ReplayBuffer, DeepNN
from env_utils import EnvWrapper
from rl_utils import calc_return

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/train', 'Directory where to write event logs and checkpoints.')

# Flags for logging
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('log_frequency', 10, 'How often to log results to the console.')

# Flags for creation of computational graph
tf.app.flags.DEFINE_list('dqn_layer_sizes', [25, 10], 'Layer sizes for the dqn.')

# Flags for environment handling
tf.app.flags.DEFINE_string('environment', 'CartPole-v1', 'The name of the openai gym environment to use')
tf.app.flags.DEFINE_boolean('render', True, 'Whether to render a display of the environment state.')

# Flags for termination criteria
tf.app.flags.DEFINE_integer('steps', 1000000, 'Max number of environment steps (potentially across multiple episodes).')

# Flags for algorithm parameters
tf.app.flags.DEFINE_float('learning_rate', 0.05, 'The learning rate (alpha) to be used.')
tf.app.flags.DEFINE_float('gamma', 1.0, 'The discount factor (gamma) to be used.')
tf.app.flags.DEFINE_float('epsilon', 1.0, 'The initial exploration rate (epsilon) to be used.')
tf.app.flags.DEFINE_float('epsilon_decay_rate', 0.0001, 'The exponential rate of decay for the exploration rate.')

tf.app.flags.DEFINE_integer('buffer_size', 1024, 'Maximum replay buffer size.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'Batch size sampled from replay buffer.')

tf.app.flags.DEFINE_integer('train_q_freq', 50, 'The number of steps before updating the action Q network.')
tf.app.flags.DEFINE_integer('target_q_update_freq', 100,
                            'The number of steps before updating target Q network from action Q network.')


class Model(object):
  def __init__(self, learning_rate, gamma, epsilon, epsilon_decay, env):
    self.global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar("global_step", self.global_step)

    self.action_step = tf.get_variable("action_step", [], dtype=tf.int64)
    tf.summary.scalar("action_step", self.action_step)

    # self.init_epsilon = tf.constant(explore_rate, dtype=tf.float32)
    self.epsilon = epsilon * tf.exp(-epsilon_decay * tf.cast(self.action_step, dtype=tf.float32))
    tf.summary.scalar("epsilon", self.epsilon)

    self.s = tf.placeholder(name='state', dtype=tf.float32, shape=(None, len(env.current_state)))
    self.a = tf.placeholder(name='action', dtype=tf.int32)
    self.r = tf.placeholder(name="reward", dtype=tf.float32, )
    self.n = tf.placeholder(name='next_state', dtype=tf.float32, shape=(None, len(env.current_state)))
    self.t = tf.placeholder(name='is_terminal', dtype=tf.float32)

    # Create target and action neural networks
    self.q_action = DeepNN(name='Q_action',
                           inputs=self.s,
                           hidden_layer_sizes=FLAGS.dqn_layer_sizes,
                           n_actions=env.action_space.n)

    self.q_target = DeepNN(name='Q_target',
                           inputs=self.n,
                           hidden_layer_sizes=FLAGS.dqn_layer_sizes,
                           n_actions=env.action_space.n,
                           trainable=False)

    # Epsilon greedy policy operation
    self._greedy_action = tf.argmax(self.q_action.action_values, axis=-1)
    self._random_action = tf.random_uniform(dtype=tf.int64, shape=[], maxval=env.action_space.n)

    # Greedy action if random n < epsilon, else random action
    self.inc_action_step = tf.assign_add(self.action_step, 1)
    with tf.control_dependencies([self.inc_action_step]):
      self.action = tf.cond(tf.greater(tf.random_uniform(dtype=tf.float32, shape=[]), self.epsilon),
                            lambda: self._greedy_action[0],
                            lambda: self._random_action)

    # Loss operation
    self.td_target = self.r + gamma * tf.reduce_max(self.q_target.action_values, axis=-1) * (1 - self.t)

    # Remove after debugging
    mask = tf.one_hot(self.a, depth=env.action_space.n, dtype=tf.bool, on_value=True, off_value=False)
    mask = tf.reshape(mask, shape=[-1, env.action_space.n])
    self.predicted_q = tf.boolean_mask(self.q_action.action_values, mask)

    self.loss = tf.losses.mean_squared_error(labels=self.td_target, predictions=self.predicted_q)
    tf.summary.scalar("loss", self.loss)

    # Training operation
    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    self.train = self.optimizer.minimize(loss=self.loss, global_step=self.global_step)

    # Create a copy operation for Q_action to Q_target
    copy_vars_ops = []

    for src, dest in zip(self.q_action.variables, self.q_target.variables):
      copy_vars_ops.append(tf.assign(dest, src))

    self.update_q_target = tf.group(copy_vars_ops, name='update_q_target')


def train(env):
  """Train for a number of steps."""
  model = Model(learning_rate=FLAGS.learning_rate,
                gamma=FLAGS.gamma,
                epsilon=FLAGS.epsilon,
                epsilon_decay=FLAGS.epsilon_decay_rate,
                env=env)
  replay_buffer = ReplayBuffer(state_dim=len(env.current_state),
                               max_size=FLAGS.buffer_size)
  episode = []

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
      action, action_step = mon_sess.raw_session().run([model.action, model.action_step],
                                                       feed_dict={model.s: [env.current_state]})
      transition = env.step(action)

      episode.append(transition)
      replay_buffer.append(transition)

      if transition.is_terminal:
        print 'Length of episode: {}'.format(len(episode))
        print 'Return: {}'.format(calc_return(episode, discount=FLAGS.gamma))
        episode = []

      if replay_buffer.full():
        if action_step % FLAGS.target_q_update_freq == 0:
          mon_sess.raw_session().run(model.update_q_target)

        if action_step % FLAGS.train_q_freq == 0:
          batch = replay_buffer.sample(FLAGS.batch_size)
          _, loss, global_step = mon_sess.run([model.train, model.loss, model.global_step],
                                              {model.s: batch.states,
                                               model.a: batch.actions,
                                               model.r: batch.rewards,
                                               model.n: batch.next_states,
                                               model.t: batch.is_terminal_indicators})


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.MakeDirs(FLAGS.train_dir)

  env = EnvWrapper(FLAGS.environment, render=FLAGS.render)
  train(env)


if __name__ == '__main__':
  tf.app.run()

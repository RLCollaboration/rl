#! /usr/bin/env python
"""A Tensorflow version of DQN (see https://deepmind.com/research/dqn/)
"""
import tensorflow as tf
import numpy as np
import time

from datetime import datetime

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
tf.app.flags.DEFINE_float('alpha', 0.05, 'The learning rate (alpha) to be used.')
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


def calculate_loss(mini_batch, Q_target, Q_action):
    # y = l_rew + self._discount * C.functions.max(self._qt.forward(l_next_obs), axis=1) * (1 - l_done)
    # q = C.functions.select_item(self._q.forward(l_obs), l_act)
    # loss = C.functions.mean(C.functions.square(y - q))

    # TODO: Tensorflow-ify this
    loss = 0
    for tr in mini_batch:
        y = tr.reward + gamma * np.argmax(Q_target(tr.next_state)) * (1 - tr.is_terminal)
        q = Q_action(tr.next_state)[tr.action]
        loss += np.square(y - q)

    # calculate the mean
    loss = loss / len(mini_batch)

    return loss


def train(env):
    """Train for a number of steps."""
    with tf.Graph().as_default():
        # Integer (Variable) that counts the number of examples that have been used up to this point in training
        global_step = tf.train.get_or_create_global_step()
        # inc_global_step = tf.assign(global_step, global_step + 1)

        # Constants
        gamma = tf.constant(FLAGS.gamma, tf.float32)

        # Variables
        alpha = tf.Variable(initial_value=FLAGS.alpha, dtype=tf.float32)

        # TODO: Create computational graph
        r = tf.placeholder(dtype=tf.float32, name="reward")
        s = tf.placeholder(name='s', dtype=tf.float32, shape=(None, len(env.current_state)))

        Q_action = DeepNN(name='Q_action',
                          inputs=s,
                          hidden_layer_sizes=FLAGS.dqn_layer_sizes,
                          n_actions=env.action_space.n)

        # TODO: Need a copy of Q_action with trainable=False
        Q_target = DeepNN(name='Q_target',
                          inputs=s,
                          hidden_layer_sizes=FLAGS.dqn_layer_sizes,
                          n_actions=env.action_space.n)

        # TODO: Define loss operation
        loss = tf.constant(0)

        # TODO: Create the training op
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        # train_op = cifar10.train(loss, global_step)

        # TODO: Create a copy operation for Q_action to Q_target
        # copy_network_op = ...


        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                # Can add additional arguments to the "run" parameters, regardless
                # of what is specified in the run method parameter list.
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.mini_batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        examples_per_sec, sec_per_batch))

        # Utility class that creates the tf session for you plus add on functionality
        # -- Different flavors of this exist
        with tf.train.MonitoredTrainingSession(

                # save/load model state
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook(), tf.train.SummarySaverHook(
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

                # TODO:  Is this needed?  (Can the monitored session handle this?)
                global_step_value = mon_sess.run(global_step)

                # Determine next action using an epsilon greedy policy based on Q(S,A)
                action = eps_greedy(env=env, Q=Q_action, epsilon=epsilon, session=mon_sess)

                # Execute action against environment and observe transition
                transition = env.step(action)

                replay_buffer.append(transition)

                if replay_buffer.full():

                    if global_step_value % FLAGS.target_q_update_freq == 0:
                        # TODO: Update the target network
                        # mon_sess.run([copy_network_op])
                        pass

                    if global_step_value % FLAGS.train_q_freq == 0:
                        # TODO: Sample minibatch from the replay buffer
                        mini_batch = replay_buffer.sample(FLAGS.mini_batch_size)

                        # TODO: Calculate loss
                        # loss = calculate_loss(mini_batch, Q_target)

                        # TODO: Update parameters
                        # _, next_action_value, global_step_value = mon_sess.run([train_op, next_action, global_step],
                        #                                                        {r: reward, s: next_state})


def main(argv=None):  # pylint: disable=unused-argument
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    env = EnvWrapper(FLAGS.environment, render=FLAGS.render)
    train(env)


if __name__ == '__main__':
    tf.app.run()

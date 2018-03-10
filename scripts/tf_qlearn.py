#! /usr/bin/env python
"""A Tensorflow version of DQN (see https://deepmind.com/research/dqn/)
"""
from datetime import datetime
import time
from env_utils import EnvWrapper
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_list('dqn_layer_sizes', [500, 100, 10],
                         """Layer sizes for the dqn.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """The name of the openai gym environment to use""")
tf.app.flags.DEFINE_boolean('render', False,
                            """Whether to render the environment in a graphical display.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, """The learning rate (eta) to be used.""")


# TODO: Need to manage the max size of replay_buffer and return a random sample of n elements (could use shuffle + copy)
replay_buffer = list()


def create_layer(scope_name, input, layer_size, activation=None, trainable=True):
    with tf.variable_scope(scope_name):
        input_size = input.get_shape()[1]
        w = tf.get_variable("weights", (input_size, layer_size), trainable=trainable, dtype=tf.float32)
        b = tf.get_variable("bias", (layer_size,), initializer=tf.zeros_initializer(),
                            trainable=trainable, dtype=tf.float32)
        output = tf.matmul(input, w) + b
        if activation:
            output = activation(output)
        return output, w, b


def create_dnn(name, input, layer_sizes, actions, trainable=True):
    with tf.variable_scope(name):
        previous = input
        variables = []
        for i, layer_size in enumerate(layer_sizes):
            previous, w, b = create_layer("Hidden_{}".format(i), previous,
                                          layer_size, activation=tf.nn.relu,
                                          trainable=trainable)
            variables += [w, b]

        action_rewards, w, b = create_layer("Final", previous,
                                            actions, activation=None,
                                            trainable=trainable)
        variables += [w, b]
        # action_distribution = tf.nn.softmax(logits)
    return action_rewards, variables


def calculate_loss(action, reward, discount, next_state, done):
    # Hint: You may want to make use of the following fields: self._discount, self._q, self._qt
    #  Hint2: Q-function can be called by self._q.forward(argument)
    #  Hint3: You might also find https://docs.chainer.org/en/stable/reference/generated/chainer.functions.select_item.html useful
    y = l_rew + self._discount * C.functions.max(self._qt.forward(l_next_obs), axis=1) * (1 - l_done)
    q = C.functions.select_item(self._q.forward(l_obs), l_act)
    loss = C.functions.mean(C.functions.square(y - q))


def train(env):
    # Counter for the number of episodes completed
    n_episodes = 0

    """Train for a number of steps."""
    with tf.Graph().as_default():
        # Integer (Variable) that counts the number of examples that have been used up to this point in training
        global_step = tf.train.get_or_create_global_step()

        # TODO: Create computational graph
        r = tf.placeholder(dtype=tf.float32, name="reward")
        s = tf.placeholder(dtype=tf.float32, name="current_state")

        create_dnn('dqn', input=s, layer_sizes=FLAGS.dqn_layer_sizes, actions=env.action_space.n)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        # logits = cifar10.inference(images)

        # TODO: Calculate loss.
        # loss = cifar10.loss(logits, labels)
        loss = 0

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        # train_op = cifar10.train(loss, global_step)


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
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
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
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook(), tf.train.SummarySaverHook(
                        save_steps=100,
                        save_secs=None,
                        output_dir=None,
                        summary_writer=None,
                        scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()),
                        summary_op=None
                    )],

                # Can be configured for multi-machine training (check out docs for this class)
                config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement)) as mon_sess:

            while not mon_sess.should_stop():

                # TODO: Determine next action using an epsilon greedy policy based on Q(S,A)
                action = env.action_space.sample()

                next_state, reward, terminal_state, _ = env.step(action)

                replay_buffer.append((curr_state, action ,reward, next_state, terminal_state))



                # _, next_action_value, global_step_value = mon_sess.run([train_op, next_action, global_step],
                #                                                        {r: reward, s: next_state})



                if replay_buffer.size() > FLAGS.min_buffer_size:

                    if global_step_value % FLAGS.target_q_update_freq == 0:
                        mon_sess.run([copy_network_op])


                    if global_step_value % FLAGS.train_q_freq == 0:
                        # TODO: Sample minibatch from the replay buffer

                        # TODO: Calculate loss

                        # TODO: Update parameters

                # Check for terminal state
                curr_state = env.reset() if terminal_state else next_state



def main(argv=None):  # pylint: disable=unused-argument
    # cifar10.maybe_download_and_extract()
    # if tf.gfile.Exists(FLAGS.train_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
    if not tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.MakeDirs(FLAGS.train_dir)

    env = EnvWrapper(FLAGS.env_name)
    train(env)


if __name__ == '__main__':
    tf.app.run()

import unittest

from timeit import timeit
from datatypes import ReplayBuffer
from env_utils import Transition


class TestReplayBuffer(unittest.TestCase):
    def test_init_buffer(self):
        MAX_SIZE = 10

        b = ReplayBuffer(max_size=MAX_SIZE)

        # Test all properties initialized
        self.assertEqual(b.max_size, MAX_SIZE)

        # Test initial buffer size == 0
        self.assertEqual(len(b), 0)

    def test_append(self):
        MAX_SIZE = 10

        b = ReplayBuffer(max_size=MAX_SIZE)

        # Test length is correct
        length = 0
        for n in xrange(MAX_SIZE):
            b.append(Transition(state=n, action=n, next_state=n, reward=n, is_terminal=False))
            length += 1
            self.assertEqual(len(b), length)

            # Verify that buffer is not full until max size is reached
            if length < MAX_SIZE:
                self.assertFalse(b.full())
            else:
                self.assertTrue(b.full())

        # test max size constraint (size does not change when appending to buffer at max size)
        b.append(Transition(state=length + 1, action=length + 1, next_state=length + 1, reward=length + 1,
                            is_terminal=False))
        self.assertEqual(len(b), MAX_SIZE)

        # test that when adding another element at buffer max capacity preserves full-ness
        self.assertTrue(b.full())

        # test is_terminal adds 0.0 when False and 1.0 when True
        b = ReplayBuffer(max_size=10)
        b.append(Transition(state=1, action=1, next_state=1, reward=1, is_terminal=False))
        s, a, n, r, is_terminal = b[0]
        self.assertEqual(is_terminal, 0.0)

        b = ReplayBuffer(max_size=10)
        b.append(Transition(state=1, action=1, next_state=1, reward=1, is_terminal=True))
        s, a, n, r, is_terminal = b[0]
        self.assertEqual(is_terminal, 1.0)

    def test_get_item(self):
        MAX_SIZE = 100
        b = ReplayBuffer(max_size=MAX_SIZE)
        for n in xrange(MAX_SIZE):
            b.append(Transition(state=n, action=n, next_state=n, reward=n, is_terminal=False))

        # Test single index to __get_item__
        for index in xrange(MAX_SIZE):
            s, a, n, r, t = b[index]

            self.assertEqual(s, index)
            self.assertEqual(a, index)
            self.assertEqual(n, index)
            self.assertEqual(r, index)
            self.assertEqual(t, 0.0)

        # Test multiple indices to __get_item__
        indexes = [i for i in xrange(10)]
        s, a, n, r, t = b[indexes]

        self.assertEqual(len(s), 10)
        self.assertEqual(len(a), 10)
        self.assertEqual(len(n), 10)
        self.assertEqual(len(r), 10)
        self.assertEqual(len(t), 10)

        for i in indexes:
            self.assertEqual(s[i], i)
            self.assertEqual(a[i], i)
            self.assertEqual(n[i], i)
            self.assertEqual(r[i], i)
            self.assertEqual(t[i], 0.0)

    def test_random_sample(self):
        b = ReplayBuffer(max_size=100)

        # Fill buffer
        for n in xrange(100):
            b.append(Transition(state=n, action=n, next_state=n, reward=n, is_terminal=False))

        # Verify sample size
        s, a, n, r, t = b.sample(sample_size=10)

        self.assertEqual(len(s), 10)
        self.assertEqual(len(a), 10)
        self.assertEqual(len(n), 10)
        self.assertEqual(len(r), 10)
        self.assertEqual(len(t), 10)

    def test_timing(self):

        BUFFER_SIZE = 1024
        SAMPLE_SIZE = 64
        NUM_TRIALS = 10000

        b = ReplayBuffer(max_size=BUFFER_SIZE)
        for n in xrange(BUFFER_SIZE):
            b.append(Transition(state=n, action=n, next_state=n, reward=n, is_terminal=False))

        def f():
            return b.sample(sample_size=SAMPLE_SIZE)

        actual_avg_time = timeit(f, number=NUM_TRIALS) / NUM_TRIALS
        max_avg_time = 0.0001  # TODO: Adjust as needed

        self.assertLess(actual_avg_time, max_avg_time)

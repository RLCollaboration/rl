import unittest

from timeit import timeit
from datatypes import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_init_buffer(self):
        MAX_SIZE = 10

        b = ReplayBuffer(max_size=MAX_SIZE)

        # Test all properties initialized
        self.assertEqual(b.max_size, MAX_SIZE)

        # Test initial buffer size == 0
        self.assertEqual(len(b), 0)

    def test_add_elements(self):
        MAX_SIZE = 10

        b = ReplayBuffer(max_size=MAX_SIZE)

        # Test length is correct
        length = 0
        for n in xrange(MAX_SIZE):
            b.append(n)
            length += 1
            self.assertEqual(len(b), length)

            # Verify that buffer is not full until max size is reached
            if length < MAX_SIZE:
                self.assertFalse(b.full())
            else:
                self.assertTrue(b.full())

        # Test all elements added to buffer exactly once
        for n in xrange(10):
            self.assertEqual(b.count(n), 1)

        # test max size constraint (size does not change when appending to buffer at max size)
        b.append(length + 1)
        self.assertEqual(len(b), MAX_SIZE)

        # test that when adding another element at buffer max capacity preserves full-ness
        self.assertTrue(b.full())

        # test FIFO (first element should be removed)
        for n in xrange(10, 20):
            b.append(n)
            self.assertEqual(b.count(n - 10), 0, msg="{} should have been removed from buffer".format(n - 10))

    def test_random_sample(self):
        b = ReplayBuffer(max_size=100)

        # Fill buffer
        for n in xrange(100):
            b.append(n)

        # Verify sample size
        sample = b.sample(size=10)
        self.assertEqual(len(sample), 10)

        # Verify buffer is unchanged
        for n in xrange(100):
            self.assertEqual(b.count(n), 1)

        # Verify all elements of sample appear in original buffer
        for n in sample:
            self.assertTrue(b.count(n) > 0)

    def test_timing(self):

        BUFFER_SIZE = 1024
        SAMPLE_SIZE = 64
        NUM_TRIALS = 10000

        b = ReplayBuffer(max_size=BUFFER_SIZE)
        for n in xrange(BUFFER_SIZE):
            b.append(n)

        def f():
            return b.sample(size=SAMPLE_SIZE)

        actual_avg_time = timeit(f, number=NUM_TRIALS) / NUM_TRIALS
        max_avg_time = 0.00005  # TODO: Adjust as needed

        self.assertLess(actual_avg_time, max_avg_time)

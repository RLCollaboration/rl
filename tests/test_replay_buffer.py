import unittest
from datatypes import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
    def test_init_queue(self):
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
        for n in xrange(10):
            b.append(n)
            length += 1
            self.assertEqual(len(b), length)

        # Test all elements added to queue exactly once
        for n in xrange(10):
            self.assertEqual(b.count(n), 1)

        # test max size constraint
        b.append(length + 1)
        self.assertEqual(len(b), MAX_SIZE)

        # test FIFO (first element should be removed)
        for n in xrange(10, 20):
            b.append(n)
            self.assertEqual(b.count(n - 10), 0, msg="{} should have been removed from queue".format(n - 10))

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

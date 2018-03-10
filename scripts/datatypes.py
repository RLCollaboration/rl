from random import sample


class ReplayBuffer(object):
    def __init__(self, initial_size=0, max_size=100):
        self.initial_size = initial_size
        self.max_size = max_size

        self._elements = [None] * self.initial_size
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
        return [self._elements[i] for i in sample(xrange(len(self)), size)]

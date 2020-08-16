import numpy.random as npr
import numpy as np


class RingBuffer(object):
    def __init__(self, N):
        self.buffer = [None] * N
        self.N = N
        self.last_index = -1
        self.index = -1

    def add(self, sample):
        self.index = (self.index + 1) % self.N
        self.last_index = min(self.last_index + 1, self.N)
        self.buffer[self.index] = sample

    #@tf.function
    def get_batch(self, batch_size):
        select_index = npr.choice(self.last_index, batch_size)
        batch = [self.buffer[i] for i in select_index]
        return batch

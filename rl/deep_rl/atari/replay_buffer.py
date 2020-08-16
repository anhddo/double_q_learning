# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import numpy as np
from collections import namedtuple, deque
import numpy.random as npr

class RingBuffer(object):
    """
    To void creating new numpy array every time append data to buffer.
    We creating a fixed size buffer [N, 84, 84] to save images. Then, new state
    will have indices [a_1, a_2, a_3, a_4], where a_i is the index that store the
    image in the buffer.
    """
    def __init__(self, N, batch_size):
        self.buffer_size = N
        self.buffer = [None] * self.buffer_size
        self.last_index = -1 
        self.index = -1 
        self.buffer_count = np.zeros(N, dtype=np.uint8)
        self.state_batch = np.zeros((batch_size, 84, 84, 4), dtype=np.uint8)
        self.next_state_batch = np.zeros((batch_size, 84, 84, 4), dtype=np.uint8)
        self.batch_size = batch_size
        

    def add(self, states, action, reward, done):
        self.index = (self.index + 1) % self.buffer_size
        self.last_index = min(self.last_index + 1, self.buffer_size)
        self.buffer[self.index] = (states, action, reward, done)

    def get_batch(self):
        select_index = npr.choice(self.last_index, self.batch_size)
        states, action, reward, done = zip(*[self.buffer[i] for i in select_index])
        for batch_idx, images in enumerate(states):
            for j  in range(4):
                self.state_batch[batch_idx, :, :, j] = images[j]
                self.next_state_batch[batch_idx, :, :, j] = images[j+1]
        action = np.stack(action)
        reward = np.array(reward, dtype=np.float32).reshape(self.batch_size, 1)
        done = np.array(done, dtype=np.float32).reshape(self.batch_size, 1)
        return self.state_batch, action, reward, self.next_state_batch, done

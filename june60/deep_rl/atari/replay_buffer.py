# ======================================================================== 
# Author: Anh Do
# Mail: anhddo93 (at) gmail (dot) com 
# Website: https://sites.google.com/view/anhddo 
# ========================================================================

import numpy as np
from collections import namedtuple, deque
import numpy.random as npr



# ====> THIS CODE IS FASTER BUT SEEM INCORRECT






class RingBuffer(object):
    """
    To void creating new numpy array every time append data to buffer.
    We creating a fixed size buffer [N, 84, 84] to save images. Then, new state
    will have indices [a_1, a_2, a_3, a_4], where a_i is the index that store the
    image in the buffer.
    """
    def __init__(self, N, batch_size):
        self.buffer = [None] * N
        self.N = N
        self.last_index = -1 
        self.index = -1 

        img_buf_size = int(N * 1.5)
        self.free_indices = deque(range(img_buf_size))
        self.image_buf = np.zeros((img_buf_size, 84, 84), dtype=np.uint8)
        self.state_batch = np.zeros((batch_size, 84, 84, 4), dtype=np.uint8)
        self.next_state_batch = np.zeros((batch_size, 84, 84, 4), dtype=np.uint8)
        self.batch_size = batch_size

    def add(self, state_indices, action, reward, done):
        self.index = (self.index + 1) % self.N
        self.last_index = min(self.last_index + 1, self.N)
        if self.buffer[self.index] is not None:
            removed_indices, _, _, removed_done = self.buffer[self.index]
            if removed_done:
                for e in removed_indices:
                    self.free_indices.append(e)
            else:
                self.free_indices.append(removed_indices[0])
        self.buffer[self.index] = (state_indices, action, reward, done)

    def insert_image(self, img):
        ind = self.free_indices.pop()
        self.image_buf[ind, ...] = img
        return ind

    def get_batch(self):
        select_index = npr.choice(self.last_index, self.batch_size)
        state_indices, action, reward, done = zip(*[self.buffer[i] for i in select_index])
        for i, indices in enumerate(state_indices):
            self.state_batch[i, ...] = self.image_buf[indices[:-1]].transpose((1, 2, 0))
            self.next_state_batch[i, ...] = self.image_buf[indices[1:]].transpose((1, 2, 0))
        action = np.stack(action)
        reward = np.array(reward, dtype=np.float32).reshape(self.batch_size, 1)
        done = np.array(done, dtype=np.float32).reshape(self.batch_size, 1)
        return self.state_batch, action, reward, self.next_state_batch, done

import numpy as np
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s1, s2, a1, a2, r, s1_next, s2_next, done):
        self.buffer.append((s1, s2, a1, a2, r, s1_next, s2_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s1, s2, a1, a2, r, s1_next, s2_next, done = zip(*batch)

        return (np.array(s1), np.array(s2),
                np.array(a1), np.array(a2),
                np.array(r),
                np.array(s1_next), np.array(s2_next),
                np.array(done))

    def __len__(self):
        return len(self.buffer)

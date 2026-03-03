import numpy as np
import random
from collections import deque
import torch

class ReplayBuffer:
    """
    Experience Replay Memory.
    Stores (state, action, reward, next_state, done) tuples.
    Random sampling breaks temporal correlations, stabilising training.
    """
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)),
        )

    def __len__(self):
        return len(self.buffer)

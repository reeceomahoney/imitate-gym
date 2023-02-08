import torch
import random


class AMPReplayBuffer:
    def __init__(self, size):
        self.obs = []
        self.size = size

    def store(self, transitions):
        self.obs += torch.split(transitions, 1)

        n = len(self.obs) - self.size
        if n > 0:
            del self.obs[:n]

    def sample(self, n):
        return random.sample(self.obs, n)

    def get_obs(self):
        return self.obs

    def __len__(self):
        return len(self.obs)

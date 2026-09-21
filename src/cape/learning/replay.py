from collections import deque
import random

class Replay:
    def __init__(self, capacity): self.values = deque(maxlen=capacity)
    def append(self, value): self.values.append(value)
    def sample(self, size): return random.sample(self.values, min(size, len(self.values)))
    def __len__(self): return len(self.values)

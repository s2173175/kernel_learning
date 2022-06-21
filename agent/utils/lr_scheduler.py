import numpy as np


class LRScheduler:

    def __init__(self, min_lr, start, decay, num_steps=20e6):

        self.min_lr = min_lr
        self.start = start
        self.decay = decay
        self.num_steps = num_steps

    def schedule(self):

        def func(progress_remaining:float)->float:

            step = int(self.num_steps - progress_remaining*self.num_steps)
            return self.min_lr + (self.start - self.min_lr) * np.exp(-self.decay * step)

        return func

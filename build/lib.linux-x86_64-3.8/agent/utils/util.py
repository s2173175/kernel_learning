import numpy as np
import matplotlib.pyplot as plt


from scipy import stats
import pandas as pd


def plot_lr_curve():
    for start in [5e-4, 1e-4, 5e-5, 1e-5]:
        _min = 1e-9
        # start = 5e-4
        total_steps = 20e6

        def decayf(step, decay):
            return (_min + (start - _min) * np.exp( -decay * step))

        x = np.arange(0, 10e6, 500, dtype=int)
        y1 = decayf(x, 4e-7)
        y2 = decayf(x, 2e-7)
        y3 = decayf(x, 6e-7)

        plt.plot(x,y1, label='4e-7')
        plt.plot(x,y2, label='2e-7')
        plt.plot(x,y3, label='6e-7')
    plt.legend()
    plt.yscale('log')
    plt.show()
    return









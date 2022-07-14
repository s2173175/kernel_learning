
from packaging import version
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb

# experiment_id = "UvIQkYHjTxWQmQoLcavGEQ"
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# # df = experiment.get_scalars()
# # df2 = experiment.get_hparams()

# # print(df)
# # print(df2)
# print([method_name for method_name in dir(experiment)
#                   if callable(getattr(experiment, method_name))])


def view_best_h_params():
    path = './results/hp_search_long_logs/hparams_table.csv'

    indexing = sorted([f'{i}_a' for i in range(1,62)])

    # bests = np.array([4,5,10,14,16,18,19,20,21,23,24,27,28,30,31,34,36,37,38,51,53,57,58])
    bests = np.array([4,20,27,34,58])
    ids = [f'{i}_a' for i in bests]

    df = pd.read_csv(path)

    df.index = indexing

    df = df.iloc[bests]

    print(df)

view_best_h_params()
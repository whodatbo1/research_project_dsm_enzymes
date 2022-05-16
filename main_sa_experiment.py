from matplotlib import pyplot as plt
import numpy as np

import main_sa


def run_exp(instance, n_times):
    x = np.arange(0, n_times)
    y = np.empty(n_times)
    for i in range(n_times):
        y[i] = main_sa.run_sa(instance)
    plt.title("distrubution of makespans of instance " + str(instance))
    plt.hist(y, bins=np.arange(y.min(), y.max() + 2))
    plt.show()


run_exp(0, 25)

import importlib

from matplotlib import pyplot as plt
import numpy as np

import main_sa
from classes.milp import FlexibleJobShop
from simulated_annealing import graph
from simulated_annealing.graph import create_graph


def run_exp(n_times, temp, nr_instances=13):
    res = np.empty(nr_instances)
    for i in range(nr_instances):
        print("instance: " + str(i))
        y = np.empty(n_times)
        for n in range(n_times):
            y[n] = main_sa.run_sa(i, temp)
        res[i] = np.min(y)
    return res




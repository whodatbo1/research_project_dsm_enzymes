import importlib
import time

from matplotlib import pyplot as plt
import numpy as np

import main_sa
from classes.milp import FlexibleJobShop
from simulated_annealing import graph
from simulated_annealing.graph import create_graph


def run_exp_s(temps, n_runs, n_neighbours, n_swaps, nr_instances=13):
    res = np.empty(nr_instances)

    for i in range(nr_instances):
        print("instance: " + str(i))
        t = time.process_time()
        y = np.empty(n_runs)
        for n in range(n_runs):
            y[n] = main_sa.run_sa_s(i, temps, n_neighbours, n_swaps)

        res[i] = np.min(y)
        elapsed_time = time.process_time() - t
        print("instance " + str(i) + " took " + str(elapsed_time) + " seconds.")
        print("make span: " + str(res[i]))
    print(res)
    return res


def run_exp_g(temps, n_runs, n_neighbours, nr_instances=13):
    res = np.empty(nr_instances)

    for i in range(nr_instances):
        print("instance: " + str(i))
        t = time.process_time()
        y = np.empty(n_runs)
        for n in range(n_runs):
            y[n] = main_sa.run_sa_g(i, temps, n_neighbours)

        res[i] = np.min(y)
        elapsed_time = time.process_time() - t
        print("instance " + str(i) + " took " + str(elapsed_time) + " seconds.")
        print("make span: " + str(res[i]))
    print(res)
    return res




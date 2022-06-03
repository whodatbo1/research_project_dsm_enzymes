import importlib
import time

from matplotlib import pyplot as plt
import numpy as np

import main_sa
from classes.milp import FlexibleJobShop
from simulated_annealing import graph
from simulated_annealing.graph import create_graph


def run_exp_g(temps, n_runs, path, name, nr_instances=13):
    res = np.empty(nr_instances)
    box_plt = []
    for i in range(nr_instances):
        print("instance: " + str(i))
        t = time.process_time()
        y = np.empty(n_runs)
        paths = []
        for n in range(n_runs):
            r = main_sa.run_sa_g(i, temps)
            y[n] = r
        # box_plt.append(y)
        # plt.savefig(path + "\\" + name + "_boxplot.png")
        res[i] = np.min(y)
        elapsed_time = time.process_time() - t
        print("instance " + str(i) + " took " + str(elapsed_time) + " seconds.")
        print("make span: " + str(res[i]))
    # for i in box_plt:
    #     plt.boxplot(i)
    # plt.ylabel("Lowest makespan found")
    # plt.xticks(range(0, nr_instances))
    # plt.xlabel("Instances")
    # plt.legend()
    return res




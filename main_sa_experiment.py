import importlib
import time

from matplotlib import pyplot as plt
import numpy as np

import main_sa
from classes.milp import FlexibleJobShop
from simulated_annealing import graph
from simulated_annealing.graph import create_graph


def run_exp_g(temps, n_runs, path, name, delta, nr_instances):
    res = np.empty(nr_instances)
    box_plt = []
    all_m = []
    run_times = []
    for i in range(n_runs):
        all_m.append([])
    temperatures = []
    for i in range(nr_instances):
        print("instance: " + str(i))
        t = time.process_time()
        y = np.empty(n_runs)
        for n in range(n_runs):
            sa = main_sa.run_sa_g(i, temps, path, delta)
            r = sa[0]
            sa1 = sa[1]
            all_m[n] = sa1
            temperatures = sa[2]
            y[n] = r
        # box_plt.append(y)
        # plt.savefig(path + "\\" + name + "_boxplot.png")
        res[i] = np.min(y)
        elapsed_time = time.process_time() - t
        run_times.append(elapsed_time)
        sorted_m = []
        for x in range(len(all_m[0])):
            sorted_m.append([])
        for x in all_m:
            for y in range(len(x)):
                sorted_m[y].append(x[y])
        # create_t_mksp_plot(i, temperatures, all_m, path)
        print("instance " + str(i) + " took " + str(elapsed_time) + " seconds.")
        print("make span: " + str(res[i]))
    return res, run_times


def create_t_mksp_plot(inst, t, m, path):
    avg_m = []
    for i in range(len(m[0])):
        avg_m.append([])
    for i in range(len(m[0])):
        for j in m:
            avg_m[i].append(j[i])
    means = []
    for i in avg_m:
        means.append(np.mean(i))
    plt.plot(t, means, marker='o',
             label="Make spans over the iterations for instance: " + str(inst))
    plt.ylabel("make-span")
    # plt.xticks(np.arange(0, max(t)))
    plt.xlabel("temperature")
    plt.legend()
    plt.savefig(path + "\\" + "make-spans_during_instance_" + str(inst) + ".png")
    plt.figure()

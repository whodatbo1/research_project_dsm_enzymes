import os
import matplotlib.pyplot as plt
import numpy as np
import time

import pandas as pd

from main_milp import milp_solve

# Runs an experiment.
# Name is the name of the folder where the results will be saved to.
# funcs_times_labels is a list of tuples (function, t, l) where function will be ran with time limit t and displayed with label l in the plot.
# functions should take two parameters: nr_instances and time
# nr_instances is the number of instances on which to run all functions
from main_sa_experiment import run_exp_g


def run_experiment(name, funcs_times_labels, nr_instances=20):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    solutions = list(map(lambda x: (x[0](nr_instances, x[1]), x[2]), funcs_times_labels))
    for solution in solutions:
        x_val = list(map(lambda x: x[0], solution[0]))
        y_val = list(map(lambda x: x[1], solution[0]))
        file.write(solution[1] + "\n")
        file.write(str(solution[0]) + "\n")
        plt.plot(x_val, y_val, marker='o', label=solution[1])
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")


def run_sa_g(name, times, temps, n_runs, deltas, nr_instances=20):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    mean = []
    data = {}
    data.update({"instance": [*range(0, nr_instances)]})
    for i in range(nr_instances):
        mean.append([])
    for i in range(times):
        for t in temps:
            print("start temperature: " + str(t))
            for n in n_runs:
                print("n times: " + str(n))
                for d in deltas:
                    y_val_sa = run_exp_g(t, n, path, name, d, nr_instances)
                    file.write("Simulated annealing, t=" + str(t) + " runs=" + str(n))
                    file.write(str(y_val_sa[0]) + "\n")
                    plt.plot(np.arange(0, nr_instances), y_val_sa[0], marker='o',
                             label="Simulated annealing, t=" + str(t) + " runs=" + str(n) + " deltaT=" + str(d))
                    for r in range(nr_instances):
                        mean[r].append(y_val_sa[0][r])
                    data.update({"make-span": y_val_sa[0]})
                    data.update({"run-time": y_val_sa[1]})

    milp_300_m9 = [18, 24, 32, 43, 53, 67, 76, 91, 102, 116, 135, 156, 180]
    milp1800_values = [22.0, 33.0, 43.0, 55.0, 71.0, 87.0, 100.0, 109.0, 128.0, 138.0, 156.0, 178.0, 203.0]
    milp900_values = [22, 33, 43, 56, 72, 87, 100, 109, 128, 138, 156, 178, 200]
    # plt.plot(np.arange(0, nr_instances), milp_300_m9, marker='o',
    #          label="MILP solver, limited to 900 seconds")
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")
    plt.figure()
    # plt.boxplot(mean, labels=np.arange(0, nr_instances))
    # plt.plot(np.arange(1, nr_instances + 1), milp900_values, marker='o',
    #          label="MILP solver, limited to 900 seconds")
    # plt.ylabel("boxplot of make-spans found")
    # plt.xlabel("Instances")
    # plt.legend()
    # plt.savefig(path + "\\" + name + "_boxplot.png")
    # plt.figure()
    df = pd.DataFrame(data)
    df.to_csv(name + 'table.csv', index=False, header=False)


#run_sa_g("sa_1_50_10", 1, [50], [10])  # file-name, T, n_runs, neighbours
# run_sa_g("sa_25_10_d0.4_20inst", 1, [25], [10], [0.4])  # file-name, T, n_runs, neighbours
#run_experiment("milp-1800_13_inst20", [(milp_solve, 1800, "MILP solver, limited to 1800 second with 20 instances")])


milp1800_20values = [22, 33, 43, 55, 71, 87, 100, 109, 128, 138, 156, 178, 192, 231, 275, 320]
SA_values = [25, 42, 60, 74, 87, 98, 117, 129, 138, 151, 166, 185, 190, 210, 225, 239, 246, 259, 273, 285]
plt.plot(np.arange(0, len(milp1800_20values)), milp1800_20values, marker='o',
         label="MILP solver, limited to 1800 seconds")
plt.plot(np.arange(0, len(SA_values)), SA_values, marker='o',
         label="SA with T=25, r=10, α=0,4")
plt.ylabel("Lowest makespan found")
plt.xticks(range(0, len(SA_values)))
plt.xlabel("Instances")
plt.legend()
plt.savefig(os.path.join("solutions/experiments") + "\\" + "Results20inst.png")
plt.figure()



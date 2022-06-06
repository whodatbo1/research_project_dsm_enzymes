import os
import matplotlib.pyplot as plt
import numpy as np
import time

from main_milp import milp_solve

# Runs an experiment.
# Name is the name of the folder where the results will be saved to.
# funcs_times_labels is a list of tuples (function, t, l) where function will be ran with time limit t and displayed with label l in the plot.
# functions should take two parameters: nr_instances and time
# nr_instances is the number of instances on which to run all functions
from main_sa_experiment import run_exp_g


def run_experiment(name, funcs_times_labels, nr_instances=13):
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


def run_sa_g(name, temps, n_runs, nr_instances=13):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    for t in temps:
        print("start temperature: " + str(t))
        for n in n_runs:
            print("n times: " + str(n))
            y_val_sa = run_exp_g(t, n, path, name)
            file.write("Simulated annealing, t=" + str(t) + " runs=" + str(n))
            file.write(str(y_val_sa) + "\n")
            plt.plot(np.arange(0, nr_instances), y_val_sa, marker='o',
                     label="Simulated annealing, t=" + str(t) + " runs=" + str(n))
    milp_values = [22.0, 33.0, 43.0, 55.0, 71.0, 87.0, 100.0, 109.0, 128.0, 138.0, 156.0, 178.0, 203.0]
    plt.plot(np.arange(0, nr_instances), milp_values, marker='o',
             label="MILP solver, limited to 1800 seconds")
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")
    plt.figure()


run_sa_g("sa_100_1-10-25-n ", [100], [1, 10, 25])  # file-name, T, n_runs, neighbours
# run_experiment("milp-1800", [(milp_solve, 1800, "MILP solver, limited to 1800 second")])


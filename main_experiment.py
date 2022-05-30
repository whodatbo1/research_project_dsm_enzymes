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
from main_sa_experiment import run_exp_s, run_exp_g


def run_milp(name, funcs_times_labels, nr_instances=13):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    solutions = list(map(lambda x: (x[0](nr_instances, x[1]), x[2]), funcs_times_labels))
    for solution in solutions:
        x_val = list(map(lambda x: x[0], solution[0]))
        y_val = list(map(lambda x: x[1], solution[0]))
        file.write(solution[1] + "\n")
        file.write(str(solution[0]) + "\n")
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")


def run_milp_and_sa(name, temps, n_times, funcs_times_labels, nr_instances=13):
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
    for t in temps:
        print("start temperature: " + str(t))
        for n in n_times:
            print("n times: " + str(n))
            y_val_sa = run_exp(n, t)
            plt.plot(np.arange(0, nr_instances), y_val_sa, marker='o',
                     label="Simulated annealing, t=" + str(t) + " n=" + str(n))
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")


def run_sa_s(name, temps, n_runs, n_neighbours, n_swaps, nr_instances=13):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    for t in temps:
        print("start temperature: " + str(t))
        for n in n_runs:
            print("n times: " + str(n))
            for neigh in n_neighbours:
                print("num of neighbours: " + str(neigh))
                for s in n_swaps:
                    print("amount of swaps = " + str(s))
                    y_val_sa = run_exp_s(t, n, neigh, s)
                    file.write("Simulated annealing, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(
                        neigh) + " swaps=" + str(s) + "\n")
                    file.write(str(y_val_sa) + "\n")
                    plt.plot(np.arange(0, nr_instances), y_val_sa, marker='o',
                             label="Simulated annealing, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(neigh) + " swaps=" + str(s))
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")


def run_sa_g(name, temps, n_runs, n_neighbours, n_swaps, nr_instances=13):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    for t in temps:
        print("start temperature: " + str(t))
        for n in n_runs:
            print("n times: " + str(n))
            for neigh in n_neighbours:
                print("num of neighbours: " + str(neigh))
                for s in n_swaps:
                    print("amount of swaps = " + str(s))
                    y_val_sa = run_exp_g(t, n, neigh, s)
                    file.write("Simulated annealing, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(
                        neigh) + " swaps=" + str(s) + "\n")
                    file.write(str(y_val_sa) + "\n")
                    plt.plot(np.arange(0, nr_instances), y_val_sa, marker='o',
                             label="Simulated annealing, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(neigh) + " swaps=" + str(s))
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")

def run_sa(name, temps, n_runs, n_neighbours, n_swaps, nr_instances=13):
    path = os.path.join("solutions/experiments")
    file = open(path + "/"
                       "" + name + ".txt", "a")
    for t in temps:
        print("start temperature: " + str(t))
        for n in n_runs:
            print("n times: " + str(n))
            for neigh in n_neighbours:
                print("num of neighbours: " + str(neigh))
                for s in n_swaps:
                    print("amount of swaps = " + str(s))
                    y_val_sa = run_exp_g(t, n, neigh, s)
                    file.write("Simulated annealing g, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(
                        neigh) + " swaps=" + str(s) + "\n")
                    file.write(str(y_val_sa) + "\n")
                    plt.plot(np.arange(0, nr_instances), y_val_sa, marker='o',
                             label="Simulated annealing, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(neigh) + " swaps=" + str(s))
                    y_val_sa = run_exp_s(t, n, neigh, s)
                    file.write("Simulated annealing s, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(
                        neigh) + " swaps=" + str(s) + "\n")
                    file.write(str(y_val_sa) + "\n")
                    plt.plot(np.arange(0, nr_instances), y_val_sa, marker='o',
                             label="Simulated annealing, t=" + str(t) + " runs=" + str(n) + " neighbours=" + str(
                                 neigh) + " swaps=" + str(s))
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")


# run_experiment("milp-30-60-180-300", [(milp_solve, 30, "MILP solver, limited to 30 seconds"), (milp_solve, 60, "MILP solver, limited to 60 seconds"), (milp_solve, 180, "MILP solver, limited to 180 seconds"), (milp_solve, 300, "MILP solver, limited to 300 seconds")])
# run_milp("milp-900", [(milp_solve, [900], "MILP solver, limited to 900 seconds")])
# run_sa("sa-10_1_1_1", [1000], [1], [100], [100])  # file-name, T, n_runs, neighbours, swaps
# run_sa("sa_graph_s-g_test", [10000], [10], [10], [10])  # file-name, T, n_runs, neighbours, swaps
# run_sa_s("sa_s_graph", [1000], [100], [10], [20])  # file-name, T, n_runs, neighbours, swaps
run_sa_g("sa_g_graph", [1000], [2], [10], [10])  # file-name, T, n_runs, neighbours, swaps

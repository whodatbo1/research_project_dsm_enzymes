import os
import matplotlib.pyplot as plt
from main_milp import milp_solve


# Runs an experiment. 
# Name is the name of the folder where the results will be saved to.
# funcs_times_labels is a list of tuples (function, t, l) where function will be ran with time limit t and displayed with label l in the plot.
# functions should take two parameters: nr_instances and time
# nr_instances is the number of instances on which to run all functions
def run_experiment(name, funcs_times_labels, nr_instances=13):
    path = os.path.join("solutions\experiments\\")
    file = open(path + "\\" + name + ".txt", "a")
    solutions = list(map(lambda x : (x[0](nr_instances, x[1]), x[2]), funcs_times_labels))
    for solution in solutions:
        x_val = list(map(lambda x : x[0], solution[0]))
        y_val = list(map(lambda x : x[1], solution[0]))
        file.write(solution[1] + "\n")
        file.write(str(solution[0]) + "\n")
        plt.plot(x_val, y_val, marker='o', label= solution[1])
    file.close()
    plt.ylabel("Lowest makespan found")
    plt.xticks(range(0, nr_instances))
    plt.xlabel("Instances")
    plt.legend()
    plt.savefig(path + "\\" + name + ".png")
    

run_experiment("milp-1-2",[(milp_solve, 1, "MILP solver, limited to 1 second"), (milp_solve, 2, "MILP solver, limited to 2 seconds")])
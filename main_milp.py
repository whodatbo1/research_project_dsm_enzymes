import importlib.util

import numpy as np

from classes.milp import FlexibleJobShop
import matplotlib.pyplot as plt
import pandas as pd

# This script reads input instances and solves
# nr_instances = 15
# for i in range(0, nr_instances):
#     fileName = 'FJSP_' + str(i)
#     spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
#     mod = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(mod)
#     alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes, machineAlternatives=
#     mod.machineAlternatives, operations=mod.operations, instance=fileName, changeOvers=mod.changeOvers, orders=mod.orders)
#     alg.build_model("solutions/milp/milp_solution_" + fileName + '.csv')
from classes.milp_utils import calculate_makespan


def run_solver(fileName, time=0):
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=
                          mod.machineAlternatives, operations=mod.operations, instance=fileName,
                          changeOvers=mod.changeOvers, orders=mod.orders)
    return alg.build_model("solutions/milp/milp_solution_" + fileName + '.csv', time)


nr_instances = 12
solution = []
for i in range(0, nr_instances):
    file_name = 'FJSP_' + str(i)
    try:
        s = run_solver(file_name, 60)
        m = calculate_makespan(s)
        solution.append(m)
    except:
        pass

plt.plot(range(0, len(solution)), solution, marker='o')
# plt.scatter(range(0, nr_instances), solution)
plt.ylabel("Lowest makespan found")
plt.xticks(range(0, len(solution)))
plt.xlabel("Instances")
plt.show()
import importlib.util
from classes.milp import FlexibleJobShop
import pandas as pd

# This script reads input instances and solves
nr_instances = 15
for i in range(0, nr_instances):
    fileName = 'FJSP_' + str(i)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes, machineAlternatives=
    mod.machineAlternatives, operations=mod.operations, instance=fileName, changeOvers=mod.changeOvers, orders=mod.orders)
    alg.build_model("solutions/milp/milp_solution_" + fileName + '.csv')


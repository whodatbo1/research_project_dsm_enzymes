import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import importlib
from ga_utils.encode_decode import *

milp_res = [(22.0), (33.0), (43.0), (56.0), (68.0), (93.0), (100.0), (115.0), (128.0), (144.0), (163.0), (207.0), (214.0)]
ga_res = [(21.0), (34.0), (50.0), (67.0), (82.0), (96.0), (112.0), (126.0), (142.0), (157.0), (174.0), (195.0), (206.0)]

plt.plot(milp_res, marker='o', label= 'MILP results, 900s limit')
plt.plot(ga_res, marker='o', label= 'GA results, population_size = 500, generations = 100')

plt.ylabel("Lowest makespan found")
# plt.xticks(range(0, nr_instances))
plt.xlabel("Instances")
plt.legend()
# plt.show()

# sched = pd.read_csv('sus_sched.csv')
# sched.sort_values(by=['Machine', 'Start'], inplace=True)
# print(sched)
# sched.sort_values(by=['Job', 'Start'], inplace=True)
# print(sched)
# array([1, 3, 7, 2, 6, 3, 7, 2, 5, 7, 0, 3, 8, 6, 8], dtype=int64), array([4, 0, 3, 2, 5, 2, 3, 1, 4, 5, 1, 3, 0, 4, 0], dtype=int64))
# array([1, 5, 7, 1, 5, 5, 7, 2, 3, 7, 0, 6, 8, 6, 8], dtype=int64), array([0, 3, 3, 1, 4, 4, 5, 0, 3, 2, 2, 4, 5, 0, 1], dtype=int64))
v1 = np.array([2, 5, 7, 0, 6, 8, 1, 3, 7, 0, 4, 8, 2, 6, 1, 3, 2, 4, 1, 5, 6, 8, 3, 7, 5, 8, 4, 7], dtype=np.int64)
v2 = np.array([5, 5, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 0, 5, 5, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)
v3 = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 0, 1], dtype=np.int64)
fileName = 'FJSP_' + str(1)
spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
instance = mod
v3 = []
for job in instance.operations:
    for i in range(len(instance.operations[job])):
        v3.append(i)
v3 = np.array(v3, dtype=np.uint64)
print(v3)
s, v1, v2 = decode_schedule_active(instance, v1, v2, v3)
s.sort_values(by=['Machine', 'Start'], inplace=True)
print(s)
print(v1)
print(v2)

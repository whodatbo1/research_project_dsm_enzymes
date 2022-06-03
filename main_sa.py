import bisect
import importlib.util

from classes import milp_utils
from classes.milp import FlexibleJobShop
import pandas as pd
import numpy as np
import random
import simulated_annealing.init_schedule as init_schedule
from classes.milp_utils import get_instance_info
from simulated_annealing import get_neighbours
from simulated_annealing.get_neighbours import create_neighbours
from simulated_annealing.graph import create_graph, create_machine_edges, get_critical_path, k_insertion, \
    get_data_frame, get_job_op

"""
Pseudo-code simulated annealing DSM optimization:

Define:
Initial schedule X(0)
Temperature T
Cooling step Td

While T > 0:
    Construct Schedule Xn in neighbourhood of X(k)
    delta = comparison between Xn and X(k) (where delta >= 0 x(k) more optimal and for delta < 0 Xn is more optimal)
    if delta > 0:
        select x(k)
    else:
        select random number r in range (0, 1)
        if r < exp(-delta/T):
            select Xn
        else:
            select X(k)
    Cooling step (decrease T with Td)

TODO:
    Schedule construction // generation mechanisms
    Schedule comparison //
    Determine Cooling rate etc.
"""

# Acknowledge Marko for this :)
# v1 contains machine assignments - v1(r) is a machine
# v2 contains operation sequence - v2(r) is a job
# v3 contains the operations numbers
"""
    Produces a valid schedule by decoding v1 and v2 into a pd.Dataframe
    representation of a schedule. This implementation does NOT lead to 
    an active schedule (i.e there might be a schedule with smaller makespan
    produced by the same input vectors).  

    Basic idea - go through the order vector (v2) sequentially and assign it
    to the correct machine starting at time (max(curr_machine_times[m], curr_job_times[j]))
"""


def feasible_schedule(inst, df):
    df.sort_values(by=["Start"], inplace=True)
    jobOps = []
    jobCompletion = []
    for i in range(len(inst.jobs)):
        jobOps.append(0)
        jobCompletion.append(0)
    lastEnzyme = []
    endingTime = []
    for i in range(len(inst.machines)):
        lastEnzyme.append("")
        endingTime.append(0)
    allops = []
    for i in inst.operations.keys():
        for j in inst.operations.get(i):
            allops.append(j)
    infeasible = False

    if len(df.values) != len(allops):
        infeasible = True

    for row in df.values:
        m = row[0]
        j = row[1]
        p = row[2]
        o = row[3]
        s = row[4]
        d = row[5]
        c = row[6]

        # check if operations of one job are performed in order
        if o != jobOps[j] or jobCompletion[j] > s:
            infeasible = True

        # check if the operation is performed on a viable machine
        if m not in inst.machineAlternatives[(j, o)]:
            infeasible = True

        # check if completion time and duration are correct
        if s + d != c or d != inst.processingTimes[(j, o, m)]:
            infeasible = True

        # check if changeOver times are fulfilled
        changeOver = 0
        if lastEnzyme[m] != "":
            changeOver = inst.changeOvers[(m, lastEnzyme[m], p)]
        if endingTime[m] + changeOver > s:
            infeasible = True

        # update trackers
        jobOps[j] += 1
        jobCompletion[j] = c
        lastEnzyme[m] = p
        endingTime[m] = c

        if infeasible:
            return False
    return True


def compare_graphs(old, new):
    return get_critical_path(old)[0] - get_critical_path(new)[0]
    # old_ms = milp_utils.calculate_makespan(get_data_frame(old, inst))
    # new_ms = milp_utils.calculate_makespan(get_data_frame(new, inst))
    # return old_ms - new_ms


def neighbourhood(inst, g, v3):
    n = []
    g_n = g
    path = get_critical_path(g)[1].copy()
    trav_path = path.copy()
    trav_path.remove(-1)
    trav_path.remove(-2)
    for v in trav_path:
        for pos in range(len(inst.machineAlternatives[get_job_op(v3, v), v3[v]])):
            g_n = k_insertion(inst, g, v3, v, pos, path)
            n.append(g_n)
    return n


def run_sa_g(instance_num, temp):
    file_name = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + file_name + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=
                          mod.machineAlternatives, operations=mod.operations, instance=file_name,
                          changeOvers=mod.changeOvers, orders=mod.orders)
    s = init_schedule.create_schedule(alg)

    g_c = create_graph(alg, s[0], s[1], s[2])
    g = create_machine_edges(alg, s[0], s[2], g_c)
    df = get_data_frame(g, alg, s[2])
    if not feasible_schedule(alg, df):
        print("Infeasible schedule: " + str(df))
        return -100
    temperature = temp
    deltaT = 0.75
    found_better = 0
    while temperature > 1:
        neighbours = neighbourhood(alg, g, s[2])
        new_g = random.choice(neighbours)
        delta = compare_graphs(g, new_g)
        if delta > 0:
            g = new_g
            found_better += 1
        else:
            r = random.uniform(0, 1)
            if r < np.exp(delta / temperature):
                g = new_g
        temperature *= deltaT

    df = get_data_frame(g, alg, s[2])
    if not feasible_schedule(alg, df):
        print("Infeasible schedule: \n" + str(df))
        return -1
    print(df)
    res = milp_utils.calculate_makespan(df)
    print("found better: " + str(found_better))
    return res

# print("graph neighbours")
# file_name = 'FJSP_' + str(0)
# spec = importlib.util.spec_from_file_location('instance', "instances/" + file_name + '.py')
# mod = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(mod)
# alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
#                       machineAlternatives=
#                       mod.machineAlternatives, operations=mod.operations, instance=file_name,
#                       changeOvers=mod.changeOvers, orders=mod.orders)
#
# s = init_schedule.create_schedule(alg)
# v3 = s[2]
# g = create_graph(alg, s[0], s[1], s[2])
# g = create_machine_edges(alg, s[0], s[2], g)
# df = get_data_frame(g, alg, v3)
# better_n = 0
# for i in range(1):
#     c_p = get_critical_path(g)
#     print(g.dummy_edges)
#     print(g.pre_edges)
#     print(g.machine_edges)
#     print(g.machines)
#     print(c_p)
#     print("\n")
#     path = c_p[1]
#     path.remove(-1)
#     path.remove(-2)
#     for v in path:
#         for x in range(len(alg.machineAlternatives[get_job_op(v3, v), v3[v]])):
#     # v = random.choice(path)
#     # x = random.randrange(0, len(alg.machineAlternatives[get_job_op(v3, v), v3[v]]))
#             g_n = k_insertion(alg, g, s[2], v, x, path)
#     print(g_n.dummy_edges)
#     print(g_n.pre_edges)
#     print(g_n.machine_edges)
#     print(g_n.machines)
#     print(str(get_critical_path(g_n)) + "\n")
#     print(get_data_frame(g_n, alg, v3))
#     if c_p[0] > get_critical_path(g_n)[0]:
#         better_n += 1
# print(better_n)

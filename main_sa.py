import bisect
import importlib.util

from classes import milp_utils
from classes.milp import FlexibleJobShop
import pandas as pd
import numpy as np
import random
import simulated_annealing.init_schedule as init_schedule
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


def decode_schedule(instance, v1, v2, v3):
    # Keep track of the max completion time for each machine after we insert an operation
    curr_machine_times = {m: 0 for m in instance.machines}

    # Keep track of the max completion time for each order after we insert an operation
    curr_job_times = {j: 0 for j in instance.orders}

    # Keep track of the enzyme of the previous operation
    prev_enzyme = {m: None for m in instance.machines}
    results = []
    # Keep track of each operation for each machine
    # machine_tasks[m] contains a list of tuples [(start, end, enzyme), ...]
    machine_tasks = {m: [(0, 0, None)] for m in instance.machines}
    # The index of the first operation for each job
    init_indices = np.where(v3 == 0)[0]
    # Keep track of the index of the last completed operation for each job
    op_index = {j: 0 for j in instance.jobs}
    i = 0
    for job in instance.jobs:
        for op in instance.operations[job]:
            # Get the current job, machine and operation
            j = v2[i]
            o = instance.operations[j][op_index[j]]
            m = v1[init_indices[j] + op_index[j]]

            op_index[j] += 1

            duration = instance.processingTimes[j, o, m]

            co_time = 0

            curr_enzyme = instance.orders[j]['product']

            # Check if there is any change-over time
            if prev_enzyme[m] is not None:
                co_time = instance.changeOvers[(m, prev_enzyme[m], curr_enzyme)]

            curr_machine_times[m] += co_time

            # We can't start the job while the machine is still busy and also
            # while the previous operation from the job is not completed
            start = max(curr_job_times[j], curr_machine_times[m])
            end = start + duration

            prev_enzyme[m] = instance.orders[j]['product']
            curr_job_times[j] = max(curr_job_times[j], end)
            curr_machine_times[m] = max(curr_machine_times[m], end)

            res = {"Machine": m, "Job": j, "Product": instance.orders[j]["product"], "Operation": o, "Start": start,
                   "Duration": instance.processingTimes[j, o, m], "Completion": end}

            results.append(res)
            bisect.insort(machine_tasks[m], (start, end, curr_enzyme))
            i += 1
    sched = pd.DataFrame(results)
    sched.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    sched.to_csv('csv_output.csv', index=False)

    return sched, v1, v2


# Compares make spans of old and new
# Returns difference in make span, when ret > 0 new is more optimal
def compare_schedules(instance, old, new):
    # print("old: " + str(old))
    # print("new: " + str(new))
    old_sched = decode_schedule(instance, np.array(old[0]), np.array(old[1]), np.array(old[2]))
    new_sched = decode_schedule(instance, np.array(new[0]), np.array(new[1]), np.array(new[2]))
    # print("old: " + str(milp_utils.calculate_makespan(old_sched[0])))
    # print("new: " + str(milp_utils.calculate_makespan(new_sched[0])))
    # print("\n")
    return milp_utils.calculate_makespan(old_sched[0]) - milp_utils.calculate_makespan(
        new_sched[0])


def compare_graphs(old, new):
    return get_critical_path(old)[0] - get_critical_path(new)[0]
    # old_ms = milp_utils.calculate_makespan(get_data_frame(old, inst))
    # new_ms = milp_utils.calculate_makespan(get_data_frame(new, inst))
    # return old_ms - new_ms


def neighbourhood(inst, g, v3, n_neigh):
    n = []
    g_n = g
    for i in range(n_neigh):
        path = get_critical_path(g)[1].copy()
        for v in path:
            for pos in range(len(inst.machineAlternatives[get_job_op(v3, v), v3[v]])):
                g_n = k_insertion(inst, g, v3, v, pos)
                n.append(g_n)
    return n


def run_sa_g(instance_num, temp, n_neigh):
    file_name = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + file_name + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=
                          mod.machineAlternatives, operations=mod.operations, instance=file_name,
                          changeOvers=mod.changeOvers, orders=mod.orders)

    s = init_schedule.create_schedule(alg)

    g = create_graph(alg, s[0], s[1], s[2])
    g = create_machine_edges(alg, s[0], s[2], g)
    temperature = temp
    deltaT = 0.75
    found_better = 0
    while temperature > 1:
        neighbours = neighbourhood(alg, g, s[2], n_neigh)
        new_g = random.choice(neighbours)
        # ms = 1000
        # for n in neighbours:
        #     n_mks = get_critical_path(n)[0]
        #     if n_mks < ms:
        #         ms = n_mks
        #         new_g = n
        delta = compare_graphs(g, new_g)
        if delta > 0:
            found_better += 1
            print("old: " + str(get_critical_path(g)))
            print("new: " + str(get_critical_path(new_g)))
            g = new_g
        else:
            r = random.uniform(0, 1)
            if r < np.exp(delta / temperature):
                g = new_g
        temperature *= deltaT
        # middle = get_critical_path(alg, g)
        # print("middle makespan: " + str(middle))
    print("better found: " + str(found_better))
    res = get_critical_path(g)[0]
    return res


def run_sa_s(instance_num, temp, n_neigh, swaps):
    file_name = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + file_name + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=
                          mod.machineAlternatives, operations=mod.operations, instance=file_name,
                          changeOvers=mod.changeOvers, orders=mod.orders)

    s = init_schedule.create_schedule(alg)

    g = create_graph(alg, s[0], s[1], s[2])
    g = create_machine_edges(alg, s[0], s[2], g)
    temperature = temp
    deltaT = 0.75
    found_better = 0
    while temperature > 1:
        neighbours = create_neighbours(s, alg, n_neigh, swaps)
        new_schedule = random.choice(neighbours)
        ms = 1000
        # for n in neighbours:
        #     decode_n = decode_schedule(alg, np.array(n[0]), np.array(n[1]), np.array(n[2]))
        #     n_mks = milp_utils.calculate_makespan(decode_n[0])
        #     if n_mks < ms:
        #         ms = n_mks
        #         new_schedule = n
        delta = compare_schedules(alg, s, new_schedule)
        if delta > 0:
            found_better += 1
            s = new_schedule
        else:
            r = random.uniform(0, 1)
            if r > np.exp(delta / temperature):
                s = new_schedule
        temperature *= deltaT

    res = decode_schedule(alg, np.array(s[0]), np.array(s[1]), np.array(s[2]))
    return milp_utils.calculate_makespan(res[0])

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
# for i in range(1):
#     c_p = get_critical_path(g)
#     # print(g.machine_edges)
#     # print(g.machines)
#     print(c_p)
#     for v in c_p[1]:
#         for pos in range(len(alg.machineAlternatives[get_job_op(v3, v), v3[v]])):
#             g_n = k_insertion(alg, g, s[2], v, pos)
#     # print(g_n.machine_edges)
#     # print(g_n.machines)
#     print(str(get_critical_path(g_n)) + "\n")
#
#

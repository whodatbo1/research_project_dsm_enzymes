import bisect
import importlib.util

from classes import milp_utils
from classes.milp import FlexibleJobShop
import pandas as pd
import numpy as np
import random
import simulated_annealing.init_schedule as init_schedule
from simulated_annealing import get_neighbours

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
    # Keep track of the max completion time for each job after we insert an operation
    curr_job_times = {j: 0 for j in instance.jobs}
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


# WARNING, THIS ALGORITHM IS STILL A DRAFT, MIGHT NEED DRASTIC CHANGES

# Compares make spans of old and new
# Returns difference in make span, when ret > 0 new is more optimal
def compare_schedules(instance, old, new):
    old_sched = decode_schedule(instance, np.array(old[0]), np.array(old[1]), np.array(old[2]))
    new_sched = decode_schedule(instance, np.array(new[0]), np.array(new[1]), np.array(new[2]))
    return milp_utils.calculate_makespan(old_sched[0]) - milp_utils.calculate_makespan(new_sched[0])  # FINISH WHEN SCHEDULE INSTANCE IS DONE!!!!


# only does stuff with the first instance for now
file_name = 'FJSP_' + str(0)
spec = importlib.util.spec_from_file_location('instance', "instances/" + file_name + '.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=
                          mod.machineAlternatives, operations=mod.operations, instance=file_name,
                          changeOvers=mod.changeOvers, orders=mod.orders)

schedule = init_schedule.create_schedule(0)
makespan = decode_schedule(alg, np.array(schedule[0]), np.array(schedule[1]), np.array(schedule[2]))
print(milp_utils.calculate_makespan(makespan[0]))

temperature = 10
deltaT = 1
while temperature > 0:  # CHECK IF THIS IS A GOOD CONDITION
    neighbours = get_neighbours.create_neighbours(schedule)  # FUNCTION DOES NOT YET EXIST
    index = random.randrange(0, len(neighbours))
    new_schedule = neighbours[index]
    if compare_schedules(alg, schedule, new_schedule) > 0:
        schedule = new_schedule
    else:
        delta = compare_schedules(alg, schedule, new_schedule)  # CHECK!!!
        r = random.uniform(0, 1)
        if r < np.exp(-delta / temperature):
            print("test")
            schedule = new_schedule
        temperature -= deltaT  # Needs checking

res = decode_schedule(alg, np.array(schedule[0]), np.array(schedule[1]), np.array(schedule[2]))
print(milp_utils.calculate_makespan(res[0]))

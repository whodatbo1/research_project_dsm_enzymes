import numpy as np
from .utils import *
import pandas as pd
from datetime import datetime
import bisect
from .help import check_valid

"""
    The purpose of this class is to provide utility functions for encoding 
    a schedule into its 2-vector representation and decoding from this
    representation to a valid active schedule.
     
    The format of the schedule
    is the same as the one produced by MILP solver, i.e. a Pandas Dataframe
    with the fields Machine, Job, Product, Operation, Start, Duration and
    Completion.
    
    The 2-vector representation is a pair (v1, v2). v1 and v2 have length equal
    to the total number of unit operations which need to be completed. The 
    full description of this representation can be found in 
    https://www.sciencedirect.com/science/article/pii/S0305054807000020
    
    There is also one extra vector - v3, which is not in the paper but is useful.
    It is common for all schedules in an instance and is an ordered list of the
    operations, so e.g. if we job 0 has 3 operations and job 1 has 2 operations,
    v3 looks like : [0, 1, 2, 0, 1]. 
    
    v1 is the machine assignment vector and v2 is the operation sequence vector.
    v1 contains machines, v2 contains jobs.
"""

"""
    Calculates the makespan of the schedule
"""
def calculate_makespan(schedule):
    comp_times = schedule["Completion"]
    return comp_times.max()


"""
    Generates a random schedule encoding
    1. Assign a random valid machine to each operation to construct v1
    2. Construct v2 by chosing a random job until we are done
    3. Return v1 and v2
"""
def generate_random_schedule_encoding(instance):
    v_length = sum([len(k) for k in instance.operations.values()])
    v = np.zeros((v_length, 2)).astype(np.uint64)

    index = 0
    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            r = np.random.choice(instance.machineAlternatives[job, i])
            v[index, 0] = r
            v[index, 1] = job
            index += 1

    counts = {j: 0 for j in instance.jobs}
    max_counts = {j: len(instance.operations[j]) for j in instance.jobs}
    index = 0
    while index < v_length:
        rnint = np.random.randint(instance.nr_jobs)
        if counts[rnint] < max_counts[rnint]:
            v[index, 1] = rnint
            counts[rnint] += 1
            index += 1

    v1 = v[:, 0]
    v2 = v[:, 1]

    return v1, v2


"""
    Generates v1 and v2 from a schedule Dataframe
"""
def encode_schedule(schedule: pd.DataFrame):
    schedule.sort_values(by='Start', inplace=True)

    v2 = schedule['Job'].to_numpy()

    schedule.sort_values(by=['Job', 'Operation'], inplace=True)
    v1 = schedule['Machine'].to_numpy()
    v3 = schedule['Operation'].to_numpy()

    return v1, v2, v3


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
    schedule = pd.DataFrame(results)
    schedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    schedule.to_csv('csv_output.csv', index=False)

    return schedule, v1, v2


# v1 contains machine assignments - v1(r) is a machine
# v2 contains operation sequence - v2(r) is a job
# v3 contains the operations numbers
"""
    Produces a valid schedule by decoding v1 and v2 into a pd.Dataframe
    representation of a schedule. This implementation leads to an  
    active schedule (i.e the schedule with the smallest makespan
    produced by the same input vectors).
"""
def decode_schedule_active(instance, v1, v2, v3):
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

            if prev_enzyme[m] is not None:
                co_time = instance.changeOvers[(m, prev_enzyme[m], curr_enzyme)]

            # Here we see if there is a gap between two operations in the machine
            # where we can schedule the current operation
            found = False
            start = -1
            end = -1

            # Go through all the gaps and check if it fits
            if len(machine_tasks[m]) >= 2:
                for k in range(1, len(machine_tasks[m])):
                    # Calculate potential change over time if we insert operation in this gap
                    pot_co_time = 0
                    enz = machine_tasks[m][k - 1][2]
                    if enz is not None:
                        pot_co_time = instance.changeOvers[(m, enz, curr_enzyme)]

                    interval_start = machine_tasks[m][k - 1][1]
                    interval_end = machine_tasks[m][k][0]

                    gap = interval_end - interval_start - pot_co_time

                    potential_start = max(curr_job_times[j], interval_start) + pot_co_time
                    second_co_time = instance.changeOvers[(m, curr_enzyme, machine_tasks[m][k][2])]
                    potential_end = potential_start + duration + second_co_time
                    if gap > duration and potential_end <= interval_end:
                        found = True
                        start = potential_start
                        end = start + duration

            # In case no gap large enough was found, do standard procedure
            if not found:
                curr_machine_times[m] += co_time
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
    schedule = pd.DataFrame(results)
    schedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    schedule.to_csv('csv_output.csv', index=False)
    # c = check_valid(schedule)
    # if not c:
    #     print(repr(v1))
    #     print(repr(v2))
    #     print(repr(v3))
    v1, v2, v3 = encode_schedule(schedule)
    return schedule, v1, v2

"""
    Generates a random schedule
"""
def generate_random_schedule(instance, v3):
    v1, v2 = generate_random_schedule_encoding(instance)
    schedule, v1, v2 = decode_schedule_active(instance, v1, v2, v3)
    return schedule, v1, v2


def generate_starting_schedules(instance_num, size):
    instance = get_instance_info(instance_num)
    v3 = []
    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            v3.append(i)
    v3 = np.array(v3, dtype=np.uint64)
    sum = 0
    min_ms = 1234567890
    min_sched = None
    min_vecs = None
    schedules = []
    for i in range(size):
        s, v1, v2 = generate_random_schedule(instance, v3)
        schedules.append(s)
        ms = calculate_makespan(s)
        sum += ms
        if ms < min_ms:
            min_ms = ms
            min_sched = s
            min_vecs = (v1, v2, v3)

    print('avg makespan of', instance_num, 'is', sum/size)
    print('min makespn', min_ms)
    # print('min schedule', min_sched)
    # print('min vecs', min_vecs)
    # min_sched.to_csv('min_csv_output.csv', index=False)
    return schedules

"""
    Generates a starting population with `size` participants
    A population is a list of tuples (makespan, schedule, v1, v2)
"""
def generate_starting_population(instance_num, size):
    instance = get_instance_info(instance_num)
    v3 = []
    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            v3.append(i)
    v3 = np.array(v3, dtype=np.uint64)

    population = []

    for i in range(size):
        v1, v2 = generate_random_schedule_encoding(instance)
        schedule, v1, v2 = decode_schedule_active(instance, v1, v2, v3)
        population.append((calculate_makespan(schedule), schedule, v1, v2))
    population = sorted(population, key=lambda sched: sched[0])
    return population


if __name__ == "__main__":
    s = datetime.now()
    generate_starting_schedules(1, 1000)
    print('time elapsed', datetime.now() - s)

import pygad
import numpy as np
from utils import *
import pandas as pd
from datetime import datetime

def calculate_makespan(schedule):
    comp_times = schedule["Completion"]
    return comp_times.max()

def generate_random_schedule_encoding(instance_num):
    instance = get_instance_info(instance_num)
    v_length = sum([len(k) for k in instance.operations.values()])
    v = np.zeros((v_length, 3)).astype(np.uint64)
    index = 0

    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            r = np.random.choice(instance.machineAlternatives[job, i])
            v[index, 0] = r
            v[index, 1] = job
            v[index, 2] = i
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
    v3 = v[:, 2]

    return v1, v2, v3

def encode_schedule(schedule):
    pass


# v1 contains machine assignments - v1(r) is a machine
# v2 contains operation sequence - v2(r) is a job
# v3 contains the jobs
def decode_schedule(instance_num, v1, v2, v3):
    instance = get_instance_info(instance_num)
    curr_machine_times = {m:0 for m in instance.machines}
    curr_job_times = {j: 0 for j in instance.jobs}
    prev_enzyme = {m:None for m in instance.machines}
    op_index = {j:0 for j in instance.jobs}
    results = []

    init_indices = np.where(v3 == 0)[0]
    i = 0
    for job in instance.jobs:
        for op in instance.operations[job]:
            j = v2[i]
            o = instance.operations[j][op_index[j]]
            m = v1[init_indices[j] + op_index[j]]
            op_index[j] += 1

            duration = instance.processingTimes[j, o, m]

            start = max(curr_job_times[j], curr_machine_times[m])

            curr_machine_times[m] += duration
            if prev_enzyme[m] is not None:
                co_time = instance.changeOvers[(m, prev_enzyme[m], instance.orders[j]['product'])]
                start += co_time
                curr_machine_times[m] += co_time
            prev_enzyme[m] = instance.orders[j]['product']

            end = start + duration

            curr_job_times[j] = end

            res = {"Machine": m, "Job": j, "Product": instance.orders[j]["product"], "Operation": o, "Start": start,
                 "Duration":
                     instance.processingTimes[j, o, m], "Completion": end}
            results.append(res)
            i += 1
    schedule = pd.DataFrame(results)
    schedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    schedule.to_csv('csv_output.csv', index=False)

    return schedule


def generate_random_schedule(instance_num):
    v1, v2, v3 = generate_random_schedule_encoding(instance_num)
    schedule = decode_schedule(instance_num, v1, v2, v3)
    return schedule


def generate_staring_population(instance_num, size):
    sum = 0
    schedules = []
    for i in range(size):
        s = generate_random_schedule(instance_num)
        schedules.append(s)
        sum += calculate_makespan(s)
    print('avg makespan of', instance_num, 'is', sum/size)
    return schedules


generate_random_schedule(0)
s = datetime.now()
generate_staring_population(12, 500)
print('time elapsed', datetime.now() - s)
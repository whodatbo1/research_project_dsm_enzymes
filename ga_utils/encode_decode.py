import pygad
import numpy as np
from utils import *
import pandas as pd


def generate_random_schedule(instance_num):
    v1 = []
    v2 = []
    v3 = []

    instance = get_instance_info(instance_num)

    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            op = instance.operations[job][i]
            r = np.random.randint(len(instance.machineAlternatives[job, i]))
            print('len', len(instance.machineAlternatives[job, op]))
            print('chosen', instance.machineAlternatives[job, op][r], op)
            v1.append(instance.machineAlternatives[job, i][r])
            v3.append(i)
    counts = {j:0 for j in instance.jobs}
    max_counts = {j:len(instance.operations[j]) for j in instance.jobs}

    while len(v1) > len(v2):
        rnint = np.random.randint(instance.nr_jobs)
        if counts[rnint] < max_counts[rnint]:
            v2.append(rnint)
    print(v1)
    print(v2)
    print(v3)
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
    results = []
    for i in range(len(v1)):
        m = v1[i]
        j = v2[i]
        o = v3[i]

        start = max(curr_job_times[j], curr_machine_times[m])
        print('ptimes', instance.processingTimes)
        end = start + instance.processingTimes[j, o, m]

        results.append(
            {"Machine": m, "Job": j, "Product": instance.orders[o]["product"], "Operation": o, "Start": start,
             "Duration":
                 instance.processingTimes[j, o, m], "Completion": end})

    schedule = pd.DataFrame(results)
    schedule.to_csv('csv_output.csv', index=False)


def calculate_makespan(schedule):
    pass

v1, v2, v3 = generate_random_schedule(0)
decode_schedule(0, v1, v2, v3)
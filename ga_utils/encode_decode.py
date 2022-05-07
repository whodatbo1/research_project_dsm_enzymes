import numpy as np
from .utils import *
import pandas as pd
from datetime import datetime
import bisect
from .help import check_valid


def calculate_makespan(schedule):
    comp_times = schedule["Completion"]
    return comp_times.max()


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
def decode_schedule(instance, v1, v2, v3):
    curr_machine_times = {m: 0 for m in instance.machines}
    curr_job_times = {j: 0 for j in instance.jobs}
    prev_enzyme = {m: None for m in instance.machines}
    op_index = {j: 0 for j in instance.jobs}
    results = []
    machine_tasks = {m: [(0, 0, None)] for m in instance.machines}
    init_indices = np.where(v3 == 0)[0]
    i = 0
    for job in instance.jobs:
        for op in instance.operations[job]:
            j = v2[i]
            o = instance.operations[j][op_index[j]]
            m = v1[init_indices[j] + op_index[j]]
            op_index[j] += 1

            duration = instance.processingTimes[j, o, m]

            co_time = 0

            curr_enzyme = instance.orders[j]['product']

            if prev_enzyme[m] is not None:
                co_time = instance.changeOvers[(m, prev_enzyme[m], curr_enzyme)]

            curr_machine_times[m] += co_time

            start = max(curr_job_times[j], curr_machine_times[m])
            end = start + duration

            prev_enzyme[m] = instance.orders[j]['product']
            curr_job_times[j] = max(curr_job_times[j], end)
            curr_machine_times[m] = max(curr_machine_times[m], end)

            res = {"Machine": m, "Job": j, "Product": instance.orders[j]["product"], "Operation": o, "Start": start,
                 "Duration": instance.processingTimes[j, o, m], "Completion": end}
            # results = pd.concat([results, res], ignore_index=True, axis=0)
            results.append(res)
            bisect.insort(machine_tasks[m], (start, end, curr_enzyme))
            # machine_tasks[m].insort((start, end))
            i += 1
    schedule = pd.DataFrame(results)
    schedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    schedule.to_csv('csv_output.csv', index=False)

    return schedule, v1, v2


# v1 contains machine assignments - v1(r) is a machine
# v2 contains operation sequence - v2(r) is a job
# v3 contains the operations numbers
def decode_schedule_active(instance, v1, v2, v3):
    curr_machine_times = {m: 0 for m in instance.machines}
    curr_job_times = {j: 0 for j in instance.jobs}
    prev_enzyme = {m: None for m in instance.machines}
    op_index = {j: 0 for j in instance.jobs}
    results = []
    machine_tasks = {m: [(0, 0, None)] for m in instance.machines}
    init_indices = np.where(v3 == 0)[0]
    i = 0
    for job in instance.jobs:
        for op in instance.operations[job]:
            j = v2[i]
            o = instance.operations[j][op_index[j]]
            m = v1[init_indices[j] + op_index[j]]
            op_index[j] += 1

            duration = instance.processingTimes[j, o, m]

            co_time = 0

            curr_enzyme = instance.orders[j]['product']

            if prev_enzyme[m] is not None:
                co_time = instance.changeOvers[(m, prev_enzyme[m], curr_enzyme)]

            found = False
            start = -1
            end = -1

            if len(machine_tasks[m]) >= 2:
                for k in range(1, len(machine_tasks[m])):
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
                        start = interval_start + pot_co_time
                        end = start + duration

            if not found:
                curr_machine_times[m] += co_time
                start = max(curr_job_times[j], curr_machine_times[m])
                end = start + duration

            prev_enzyme[m] = instance.orders[j]['product']
            curr_job_times[j] = max(curr_job_times[j], end)
            curr_machine_times[m] = max(curr_machine_times[m], end)

            res = {"Machine": m, "Job": j, "Product": instance.orders[j]["product"], "Operation": o, "Start": start,
                   "Duration": instance.processingTimes[j, o, m], "Completion": end}
            # results = pd.concat([results, res], ignore_index=True, axis=0)
            results.append(res)
            bisect.insort(machine_tasks[m], (start, end, curr_enzyme))
            # print(res)
            # machine_tasks[m].insort((start, end))
            i += 1
    schedule = pd.DataFrame(results)
    schedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    schedule.to_csv('csv_output.csv', index=False)
    c = check_valid(schedule)
    if not c:
        print(repr(v1))
        print(repr(v2))
        print(repr(v3))
    v1, v2, v3 = encode_schedule(schedule)
    return schedule, v1, v2


def generate_random_schedule(instance, v3):
    v1, v2 = generate_random_schedule_encoding(instance)
    # schedule, v1, v2 = decode_schedule(instance, v1, v2, v3)
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
    # print(population[0][0])
    # print(population[0][1].sort_values(by=["Machine", "Start"]))
    # print(population[0][2])
    # print(population[0][3])
    return population


if __name__ == "__main__":
    s = datetime.now()
    generate_starting_schedules(1, 1000)
    print('time elapsed', datetime.now() - s)

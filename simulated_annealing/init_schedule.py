"""
Function: this file create the first instance of a schedule for the simulated annealing optimization for the  FJSP
Here the initial schedule will be created with global selection as described by Yang S, Guohui Z, Liang G et al.

Pseudocode:

1. Set time array with length M and initial values 0
2. Select random job, make sure it's chosen once, and select the first operation
3. get machines for the specific operation, and add the occupation time to a temp array, based on the time array
4. compare times in temp array, and get index of lowest, if 2 indexes have same time, random selection
5. add selected machine occupation time to the time array as update
6. repeat step 3-5 for all operation in the job
7. repeat step 2-6 for all jobs.

"""

import importlib.util
import numpy as np
from classes.milp import FlexibleJobShop
import pandas as pd
import random


def create_schedule(alg):
    time = np.zeros(len(alg.machines))
    num_list = list(np.arange(0, len(alg.jobs)))
    v1_2d = []
    for i in range(len(alg.jobs)):
        v1_2d.append([])
    v2 = []
    # variable to keep track of the previous job for change-over times
    prev_job = -1
    # create a loop of all order indices, if list is empty, all orders are scheduled.
    while len(num_list) != 0:
        # Randomly select an order containing a job.
        job_index = num_list.pop(random.randrange(len(num_list)))
        job = job_index
        # Get operations for the selected job
        operations = alg.operations[job]
        total_time = 0
        # For each operation, look for the lowest production time, and add that to the time array
        for op in operations:
            v2.append(job)
            temp = time.copy()
            # Get all machines and retrieve machine with lowest production time + cleaning time
            machines = alg.machineAlternatives[job, op]
            for m in machines:
                p_time = alg.processingTimes[job, op, m]
                # determine change-over
                if prev_job >= 0:
                    change_over = alg.changeOvers[m, alg.orders[prev_job].get("product"), alg.orders[job].get("product")]
                    p_time += change_over
                temp[m] += p_time + total_time
            min = float('inf')  # large number
            smallest_index = -1
            # find smallest production time, and add to time array.
            for i in range(len(temp)):
                if 0 < temp[i] <= min and (i in machines):
                    # If the smallest value is the same, randomly select one of the equal values
                    if temp[i] == min:
                        if random.random() >= 0.5:
                            min = temp[i]
                            smallest_index = i
                    else:
                        min = temp[i]
                        smallest_index = i
            # Update smallest value in array
            v1_2d[job].append(smallest_index)
            time[smallest_index] += min
            total_time += min

        # info for changeover
        prev_job = job
    # Flatten the 2d representation of vector v1
    v1 = []
    for i in v1_2d:
        for j in i:
            v1.append(j)
    # Create the 3rd vector needed for decoding
    v3 = []
    for i in alg.jobs:
        for j in range(len(alg.operations[i])):
            v3.append(j)
    return v1, v2, v3

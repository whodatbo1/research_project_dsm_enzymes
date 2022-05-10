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


def create_schedule(instance_num):
    # get instance
    file_name = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + file_name + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    alg = FlexibleJobShop(jobs=mod.jobs, machines=mod.machines, processingTimes=mod.processingTimes,
                          machineAlternatives=
                          mod.machineAlternatives, operations=mod.operations, instance=file_name,
                          changeOvers=mod.changeOvers, orders=mod.orders)
    # Create array for the time, a list of the indices of orders and an array containing al orders
    time = np.zeros(len(alg.machines))
    num_list = list(np.arange(0, len(alg.orders)))
    orders = alg.orders
    # variable to keep track of the previous job for change-over times
    prev_job = -1
    # create a loop of all order indices, if list is empty, all orders are scheduled.
    while len(num_list) != 0:
        # copy the time array onto a temp array, to make that is is not changed unintentionally.

        # Randomly select an order containing a job.
        job = -1
        order_index = num_list.pop(random.randrange(len(num_list)))
        for i in range(len(alg.jobs)):
            order = orders[order_index].get('product')
            if order == 'enzyme' + str(i):
                job = i
        # Get operations for the selected job
        operations = alg.operations[job]
        total_time = 0
        # For each operation, look for the lowest production time, and add that to the time array
        for op in operations:
            temp = time.copy()
            # Get all machines and retrieve machine with lowest production time + cleaning time
            machines = alg.machineAlternatives[job, op]
            print(machines)
            for m in machines:
                p_time = alg.processingTimes[job, op, m]
                # determine change-over
                if prev_job >= 0:
                    change_over = alg.changeOvers[m, 'enzyme' + str(prev_job), 'enzyme' + str(job)]
                    p_time += change_over
                print(p_time)
                temp[m] += p_time + total_time
            min = 1000  # large number
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
            time[smallest_index] += min
            print(temp)
            print(total_time)
            print(time)
            print("\n")
            total_time += min


        # info for changeover
        prev_job = job

    return time

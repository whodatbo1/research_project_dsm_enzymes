import random

import numpy as np

"""
Create the most beautiful pseudocode ever for neighbouring here
uses graphs

"""


def create_neighbours(schedule, alg, n_neigh, swaps):
    neighbours = []
    for i in range(n_neigh):
        v1 = schedule[0].copy()
        v2 = schedule[1].copy()
        v3 = schedule[2].copy()
        neighbour = [change_machine(v1, v3, alg), swap_operations(v2, alg, swaps), v3]
        neighbours.append(neighbour)
    return neighbours


# change machines that the operations are run on
def change_machine(v1, v3, alg):
    # print("old v1: " + str(v1))
    job = -1
    for i in range(len(v3)):
        if v3[i] == 0:
            job += 1
        op = v3[i]
        machines = alg.machineAlternatives[job, op]
        r = random.random()
        if r >= 0.5:
            v1[i] = random.choice(machines)
    # print("new v1: " + str(v1))
    return v1


# randomly swap operations in the operation vector
def swap_operations(v2, alg, swaps):
    for i in range(swaps):
        s1 = random.randrange(len(v2))
        s2 = random.randrange(len(v2))
        temp = v2[s1]
        v2[s1] = v2[s2]
        v2[s2] = temp
    return v2

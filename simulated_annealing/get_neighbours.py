import random

import numpy as np

"""
Create the most beautiful pseudocode ever for neighbouring here


"""


def create_neighbours(schedule):
    amount_neighbours = random.randrange(1, 3)
    neighbours = []
    for i in range(amount_neighbours):
        v1 = schedule[0].copy()
        v2 = schedule[1].copy()
        v3 = schedule[2].copy()
        neighbour = [change_machine(v1), swap_operations(v2), v3]
        neighbours.append(neighbour)
    return neighbours


def change_machine(v1):
    return v1


def swap_operations(v2):
    swaps = random.randrange(10, 20)
    for i in range(swaps):
        s1 = random.randrange(len(v2))
        s2 = random.randrange(len(v2))
        temp = v2[s1]
        v2[s1] = v2[s2]
        v2[s2] = temp
    return v2

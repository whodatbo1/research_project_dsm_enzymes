import random

import numpy as np

"""
Create the most beautiful pseudocode ever for neighbouring here


"""


def create_neighbours(schedule):
    amount_neighbours = random.randrange(10, 20)
    neighbours = np.array(amount_neighbours)
    for i in range(amount_neighbours):
        neighbours[i] = [change_machine(schedule[0]), swap_operations(schedule[1])]
    return neighbours


def change_machine(v1):
    return v1


def swap_operations(v2):
    swaps = random.randrange(1, 5)
    for i in range(swaps):
        num_list = list(np.arange(0, len(v2)))
        s1 = num_list.pop(random.randrange(len(num_list)))
        s2 = num_list.pop(random.randrange(len(num_list)))
        temp = v2[1][s1]
        v2[s1] = v2[1][s2]
        v2[s2] = temp
    return v2

import importlib.util
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


# WARNING, THIS ALGORITHM IS STILL A DRAFT, MIGHT NEED DRASTIC CHANGES

# Compares make spans of old and new
# Returns difference in make span, when ret > 0 new is more optimal
def compare_schedules(old, new):
    return old - new  # FINISH WHEN SCHEDULE INSTANCE IS DONE!!!!


schedule = init_schedule.create_schedule(0)
print(schedule)
temperature = 10
deltaT = 1
while temperature > 0:  # CHECK IF THIS IS A GOOD CONDITION
    neighbours = get_neighbours.create_neighbours(schedule)  # FUNCTION DOES NOT YET EXIST
    index = random.randrange(0, len(neighbours))
    new_schedule = neighbours[index]
    if compare_schedules(schedule, new_schedule) > 0:
        schedule = new_schedule
    else:
        delta = compare_schedules(schedule, new_schedule)  # CHECK!!!
        r = random.uniform(0, 1)
        if r < np.exp(-delta / temperature):
            schedule = new_schedule
    temperature -= deltaT # Needs checking

print(schedule)

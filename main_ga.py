import pygad
import numpy as np
import importlib.util
import ga_utils.utils
import ga_utils.encode_decode

def fitness_function(schedule):
    v1, v2 = schedule

# There are 2 types of mutations
# 1. Any machine assignment can be swapped for another machine (v1)
# 2. Any order of elements can be swapped in the order vector (v2)
def mutate_schedule(instance, encoded_schedule, mutation_coef: float):
    v1, v2, v3 = encoded_schedule
    op_index = {j: 0 for j in instance.jobs}

    for i in range(len(v1)):
        # Determine if we perform mutation type 1 on gene
        if np.random.rand() < mutation_coef:
            j = v2[i]
            op = instance.operations[j][op_index[j]]
            op_index[j] += 1
            new_m = np.random.choice(instance.machineAlternatives[j, op])
            v1[i] = new_m
        # Determine if we perform mutation type 2 on gene
        elif np.random.rand() < mutation_coef:
            coef = np.random.randint(0, len(v1))
            placeholder = v1[coef]
            v1[coef] = v1[i]
            v1[i] = v1[coef]
        else:
            pass


def crossover(instance, s1, s2):
    pass

def pipeline(instance_num, size, generations):
    population = generate_starting_population(instance_num, size)
    for i in range(generations):


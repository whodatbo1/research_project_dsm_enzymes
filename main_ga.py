import pygad
import numpy as np
import importlib.util
from ga_utils.utils import get_instance_info
from ga_utils.encode_decode import *
import pickle


def fitness_function(schedule):
    return calculate_makespan(schedule)


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
            v1[i] = placeholder
        else:
            pass


def get_new_rep_v2(jobs, v2):
    new_rep = np.zeros_like(v2)
    curr_i = 0

    for i in jobs:
        indices = np.where(v2 == i)[0]
        # print(i, '1', indices)
        new_rep[curr_i: curr_i + len(indices)] = indices

        curr_i += len(indices)

    return new_rep


def get_v2_from_new_rep(instance, new_rep_v2):
    job_vector = np.zeros(len(new_rep_v2), dtype=np.int64)
    index = 0
    for job in instance.jobs:
        op_count = len(instance.operations[job])
        # print(job, 'aaaaaaa', job_vector[index:(index + op_count)], index)
        job_vector[index:(index + op_count)] = np.full(op_count, job)
        index = index + op_count
    # print('jv', job_vector)
    # print(new_rep_v2)
    v2 = [job_vector[i] for i in new_rep_v2]

    return v2


def crossover(instance, schedule_male, schedule_female):
    v1_male, v2_male = schedule_male
    v1_female, v2_female = schedule_female

    # print(v2_female)
    # print(v2_male)

    length = len(v1_male)

    new_rep_male = get_new_rep_v2(instance.jobs, v2_male)
    new_rep_female = get_new_rep_v2(instance.jobs, v2_female)

    print('v1_male\n', v1_male)
    print('v1_female\n', v1_female)
    print('nrm\n', new_rep_male)
    print('nrf\n', new_rep_female)

    v1_child = np.full(length, -1, dtype=np.int64)
    v2_child = np.full(length, -1, dtype=np.int64)
    v3_child = np.full(length, -1, dtype=np.int64)

    indices = sorted(np.random.randint(0, len(v1_male), 2))
    i1, i2 = indices
    new_rep_child = np.full(length, -1, dtype=np.int64)
    new_rep_child[i1:i2] = new_rep_male[i1:i2]
    v1_child[i1:i2] = v1_male[i1:i2]

    print(new_rep_child)

    available_indices = np.append(np.arange(0, i1), np.arange(i2, length))
    index = 0
    for i in range(length):
        if new_rep_female[i] not in new_rep_child:
            new_rep_child[available_indices[index]] = new_rep_female[i]
            v1_child[available_indices[index]] = v1_female[i]
            index += 1

    print(i1, i2)
    print(new_rep_child)
    print('v1_child\n', v1_child)
    v2_child = get_v2_from_new_rep(instance, new_rep_child)
    print(v2_child)
    return v1_child, v2_child


def pipeline(instance_num, size, generations, fitness):
    fileName = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    instance = mod
    print('Generating starting population...')
    population = generate_starting_population(instance_num, size)

    probability_vector = np.ones(len(population[0][2]), dtype=np.float64)
    roulette_size = int(len(population[0][2]) * 9 / 10)
    print(roulette_size)
    print('kek', np.linspace(0, 1, roulette_size))
    probability_vector[len(population[0][2]) - roulette_size:] = np.linspace(0, 1, roulette_size)

    probability_vector = probability_vector / np.sum(probability_vector)
    print('pv', probability_vector)

    parent_count = int(size / 2)
    if parent_count % 2 == 1:
        parent_count += 1

    v3 = []
    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            v3.append(i)
    v3 = np.array(v3, dtype=np.uint64)

    for i in range(generations):
        parents = np.random.choice(len(population[0][2]), parent_count, p=probability_vector)
        for i in np.arange(0, len(parents), 2):
            # print(fitness_value)
            p1 = population[parents[i]]
            p2 = population[parents[i + 1]]
            p1_vectors = (p1[2], p1[3])
            p2_vectors = (p2[2], p2[3])
            v1_child, v2_child = crossover(instance, p1_vectors, p2_vectors)
            mutate_schedule(instance, (v1_child, v2_child, v3), 0.1)
            schedule, v1, v2 = decode_schedule_active(instance, v1_child, v2_child, v3)
            population.append((calculate_makespan(schedule), schedule, v1, v2))

        population = sorted(population, key=lambda sched: sched[0])[:size]
        print('Min makespan gen', i, population[0][0])


def run():
    # fileName = 'FJSP_' + str(0)
    # spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    # mod = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(mod)
    # instance = mod
    #
    # try:
    #     with open(r"parents.pickle", "rb") as output_file:
    #         sm, sf = pickle.load(output_file)
    #         # print(contents)
    #         print('read from dump')
    #         crossover(instance, sm, sf)
    #     return
    # except (OSError, EOFError) as e:
    #     print('generate new parents')
    #     sm = generate_random_schedule_encoding(instance)
    #     sf = generate_random_schedule_encoding(instance)
    #
    #     parents = (sm, sf)
    #
    #     with open(r"parents.pickle", "wb") as output_file:
    #         pickle.dump(parents, output_file)
    #
    #     crossover(instance, sm, sf)
    print("Starting GA...")

    pipeline(0, 100, 100, fitness_function)


if __name__ == "__main__":
    run()

import numpy as np
import importlib.util
from ga_utils.encode_decode import *
import pickle
from datetime import datetime
import bisect
from heapq import merge


def fitness_function(schedule):
    return calculate_makespan(schedule)


def get_instance_info(instance_num: int):
    fileName = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# There are 2 types of mutations
# 1. Any machine assignment can be swapped for another machine (v1)
# 2. Any order of elements can be swapped in the order vector (v2)
def mutate_schedule(instance, encoded_schedule, job_vector, mutation_coef: float):
    v1, v2, v3 = encoded_schedule

    op_index = {job: 0 for job in instance.jobs}
    for i in range(len(v1)):
        # Determine if we perform mutation type 1 on gene
        j = job_vector[i]
        op = instance.operations[j][op_index[j]]
        op_index[j] += 1
        if np.random.rand() < mutation_coef:
            new_m = np.random.choice(instance.machineAlternatives[j, op])
            v1[i] = new_m
        # Determine if we perform mutation type 2 on gene
        elif np.random.rand() < mutation_coef:
            coef = np.random.randint(0, len(v2))
            placeholder = v2[i]
            v2 = v2.tolist()
            del v2[i]
            v2.insert(coef, placeholder)
            v2 = np.array(v2)
        else:
            pass

    return v1, v2


def get_new_rep_v2(jobs, v2):
    new_rep = np.zeros_like(v2)
    curr_i = 0

    for i in jobs:
        indices = np.where(v2 == i)[0]
        new_rep[curr_i: curr_i + len(indices)] = indices

        curr_i += len(indices)

    return new_rep


def get_v2_from_new_rep(job_vector, new_rep_v2):
    v2 = [job_vector[i] for i in new_rep_v2]

    return np.array(v2)


def crossover(instance, schedule_male, schedule_female, job_vector):
    v1_male, v2_male = schedule_male
    v1_female, v2_female = schedule_female

    length = len(v1_male)

    new_rep_male = get_new_rep_v2(instance.jobs, v2_male)
    new_rep_female = get_new_rep_v2(instance.jobs, v2_female)

    v1_child = np.full(length, -1, dtype=np.int64)

    i1 = np.random.randint(0, int(len(v1_male) * 0.4))
    i2 = min(i1 + int(len(v1_male) * 0.6), len(v1_male - 1))

    new_rep_child = np.full(length, -1, dtype=np.int64)
    new_rep_child[i1:i2] = new_rep_male[i1:i2]

    for ind in new_rep_child[i1:i2]:
        v1_child[ind] = v1_male[ind]

    available_indices = np.append(np.arange(0, i1), np.arange(i2, length))
    index = 0
    for i in range(length):
        if new_rep_female[i] not in new_rep_child:
            new_rep_child[available_indices[index]] = new_rep_female[i]
            v1_child[new_rep_female[i]] = v1_female[new_rep_female[i]]
            index += 1

    v2_child = get_v2_from_new_rep(job_vector, new_rep_child)
    return v1_child, v2_child, i1, i2


# Initializes some parameters before we run the GA
def initialize_pipeline(size, instance):
    parent_count = int(size / 2)
    if parent_count % 2 == 1:
        parent_count += 1

    # Construct v3
    v3 = []
    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            v3.append(i)
    v3 = np.array(v3, dtype=np.uint64)

    # Construct job_vector
    # For each operation in a job,
    job_vector = np.zeros(len(v3), dtype=np.int64)
    index = 0
    for job in instance.jobs:
        op_count = len(instance.operations[job])
        job_vector[index:(index + op_count)] = np.full(op_count, job)
        index = index + op_count

    return parent_count, v3, job_vector


# Main GA pipeline
def pipeline(instance_num, size, generations, fitness):
    start_time = datetime.now()

    instance = get_instance_info(instance_num)

    print('Starting pipeline with instance', str(instance_num) + ', population size', str(size) + ' and max generation count', str(generations) + '...')
    print('Generating starting population...')

    population = generate_starting_population(instance_num, size)

    parent_count, v3, job_vector = initialize_pipeline(size, instance)

    tie_breaker = -size - 1

    # For each generation we store the Minimum Makespan and the Average Makespan
    results = {}

    # Perform GA
    for gen in range(generations):
        probability_vector = np.array(population)[:, 0].astype(np.float64)
        probability_vector = (np.max(probability_vector) - probability_vector) / (
                    np.max(probability_vector) - np.min(probability_vector))
        probability_vector = probability_vector / np.sum(probability_vector)

        print("Generation", gen + 1, '...')

        # Timing
        gen_start = datetime.now()

        # Selection
        parents = np.random.choice(size, parent_count, p=probability_vector)

        elite_size = int(size/10)
        non_elite_size = size - elite_size
        elites = population[:elite_size]
        non_elites = population[elite_size:]
        children = []

        for i in np.arange(0, len(parents), 2):
            p1 = population[parents[i]]
            p2 = population[parents[i + 1]]
            p1_vectors = (p1[3], p1[4])
            p2_vectors = (p2[3], p2[4])

            # Crossover
            v1_child, v2_child, i1, i2 = crossover(instance, p1_vectors, p2_vectors, job_vector)

            schedule, v1, v2 = decode_schedule_active(instance, v1_child, v2_child, v3)

            bisect.insort(children, (fitness(schedule), tie_breaker, schedule, v1_child, v2_child))

            tie_breaker -= 1

        # Mutation
        for i in range(len(non_elites)):
            fitness_value, tie_breaker_mutation, schedule, v1, v2 = non_elites[i]
            v1, v2 = mutate_schedule(instance, (v1, v2, v3), job_vector, 0.2)
            non_elites[i] = (fitness_value, tie_breaker_mutation, schedule, v1, v2)

        for i in range(len(children)):
            fitness_value, tie_breaker_mutation, schedule, v1, v2 = children[i]
            v1, v2 = mutate_schedule(instance, (v1, v2, v3), job_vector, 0.2)
            children[i] = (fitness_value, tie_breaker_mutation, schedule, v1, v2)

        next_gen_population = list(merge(children, non_elites))[:non_elite_size]

        population = list(merge(next_gen_population, elites))

        makespan_sum = 0
        for i in population:
            makespan_sum += i[0]
        average_makespan = makespan_sum/size
        minimum_makespan = population[0][0]

        print('Minimum makespan gen', gen + 1, minimum_makespan)
        print('Total time elapsed:', datetime.now() - gen_start)
        print('Average makespan gen', gen + 1, average_makespan)

        # results.append((size, generations, gen, population[0][0], sum/size))
        results[gen] = {'min': minimum_makespan, 'avg': average_makespan}

    print('Total time elapsed:', datetime.now() - start_time)
    print(population[0])
    return results


def read_static():
    fileName = 'FJSP_' + str(0)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    instance = mod

    try:
        with open(r"parents.pickle", "rb") as output_file:
            sm, sf = pickle.load(output_file)
            # print(contents)
            print('read from dump')
            crossover(instance, sm, sf)
        return
    except (OSError, EOFError) as e:
        print('generate new parents')
        sm = generate_random_schedule_encoding(instance)
        sf = generate_random_schedule_encoding(instance)

        parents = (sm, sf)

        with open(r"parents.pickle", "wb") as output_file:
            pickle.dump(parents, output_file)

        schedule, v1, v2 = crossover(instance, sm, sf)
        v3 = []
        for job in instance.operations:
            for i in range(len(instance.operations[job])):
                v3.append(i)
        v3 = np.array(v3, dtype=np.uint64)
        decode_schedule_active(instance, v1, v2, v3)


def run():
    print("Starting GA...")
    results = {}
    # Generation count, Population size, makespan
    gen_count = [100, 200, 300, 400, 500]
    pop_size = [50, 100]
    gen_count = [100]
    # pop_size = [50]
    for s in pop_size:
        for g in gen_count:
            results[(s, g)] = pipeline(1, s, g, fitness_function)
    with open('res.csv', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    run()


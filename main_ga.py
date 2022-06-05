import pickle
from datetime import datetime
import bisect
from heapq import merge
from main.ga_classes.instance import Instance
from main.ga_classes.schedule import Schedule, calculate_latency, calculate_makespan, calculate_total_machine_workload, calculate_max_machine_workload
from main.ga_utils.utils import get_instance_info
from main.ga_utils.encode_decode import *


# Generates a starting population with `size` participants
# A population is a list of tuples (makespan, tie_breaker, Schedule)
def generate_starting_population_zhang(instance, size) -> [(int, int, Schedule)]:
    population = []

    for i in range(size):
        schedule = Schedule(instance)
        v1, v2 = instance.generate_random_schedule_encoding_zhang()
        schedule.construct_from_vectors(v1, v2)
        population.append((schedule.makespan, -i, schedule))

    population = sorted(population, key=lambda sched: (sched[0], sched[1]))
    return population


def generate_starting_population_random(instance, size) -> [(int, int, Schedule)]:
    population = []

    for i in range(size):
        schedule = Schedule(instance)
        v1, v2 = instance.generate_random_schedule_encoding()
        schedule.construct_from_vectors(v1, v2)
        population.append((schedule.makespan, -i, schedule))

    population = sorted(population, key=lambda sched: (sched[0], sched[1]))
    return population


def fitness_function(schedule: Schedule):
    return calculate_makespan(schedule)


# There are 2 types of mutations
# 1. Any machine assignment can be swapped for another machine (v1)
# 2. Any order of elements can be swapped in the order vector (v2)
def mutate_schedule(instance: Instance, schedule: Schedule, mutation_coef: float):
    job_vector = instance.job_vector.copy()
    op_index = {job: 0 for job in instance.jobs}
    for i in range(instance.vector_length):
        # Determine if we perform mutation type 1 on gene
        j = job_vector[i]
        op = instance.operations[j][op_index[j]]
        op_index[j] += 1
        if np.random.rand() < mutation_coef:
            new_m = np.random.choice(instance.machine_alternatives[j, op])
            schedule.v1[i] = new_m
        # Determine if we perform mutation type 2 on gene
        elif np.random.rand() < mutation_coef:
            coef = np.random.randint(0, instance.vector_length)
            placeholder = schedule.v2[i]
            v2 = schedule.v2.tolist()
            del v2[i]
            v2.insert(coef, placeholder)
            schedule.v2 = np.array(v2)
        else:
            pass


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


# Main GA pipeline
def pipeline(instance_num, size, generations, fitness):
    start_time = datetime.now()

    instance = get_instance_info(instance_num)

    job_vector = instance.job_vector
    parent_count = int(size / 2)
    if parent_count % 2 == 1:
        parent_count += 1

    print('Starting pipeline with instance', str(instance_num) + ', population size', str(size) + ' and max generation count', str(generations) + '...')
    print('Generating starting population...')

    population = generate_starting_population_zhang(instance, size)

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
            p1_vectors = (p1[2].v1, p1[2].v2)
            p2_vectors = (p2[2].v1, p2[2].v2)

            # Crossover
            v1_child, v2_child, i1, i2 = crossover(instance, p1_vectors, p2_vectors, job_vector)

            schedule = Schedule(instance).construct_from_vectors(v1_child, v2_child)
            calculate_latency(schedule)
            calculate_total_machine_workload(schedule)
            calculate_max_machine_workload(schedule)
            bisect.insort(children, (fitness(schedule), tie_breaker, schedule))

            tie_breaker -= 1

        # Mutation
        for i in range(len(non_elites)):
            _, _, schedule = non_elites[i]
            mutate_schedule(instance, schedule, 0.2)

        for i in range(len(children)):
            _, _, schedule = children[i]
            mutate_schedule(instance, schedule, 0.2)

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

        results[gen] = {'min': minimum_makespan, 'avg': average_makespan}

    print('Total time elapsed:', datetime.now() - start_time)
    print(population[0])
    return results


def run():
    print("Starting GA...")
    results = {}
    pop_size = [50, 100]
    gen_count = [100]
    for s in pop_size:
        for g in gen_count:
            results[(s, g)] = pipeline(1, s, g, fitness_function)
    with open('res.csv', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    run()


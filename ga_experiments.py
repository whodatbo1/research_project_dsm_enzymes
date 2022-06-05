import pickle
import time
from main.main_ga import pipeline, fitness_function


def obtain_convergence_plot_data(instance_num: int):
    population_sizes = [50]
    generation_counts = [100]
    results = {}
    for size in population_sizes:
        for generation_count in generation_counts:
            results[(size, generation_count)] = pipeline(instance_num, size, generation_count, fitness_function)
    timestr = time.strftime("%Y_%m_%d_%H-%M-%S")
    with open('./ga_results/convergence_results_' + timestr + '.csv', 'wb') as f:
        pickle.dump(results, f)


def obtain_data_from_multiple_runs(instance_num: int, population_size: int, generation_count: int, runs: int):
    results = {}
    for run in range(runs):
        results[(population_size, generation_count, run)] = pipeline(instance_num, population_size, generation_count, fitness_function)
    timestr = time.strftime("%Y_%m_%d_%H-%M-%S")
    with open('./ga_results/multirun_results_' + timestr + '.csv', 'wb') as f:
        pickle.dump(results, f)


# obtain_convergence_plot_data(1)
obtain_data_from_multiple_runs(12, 100, 100, 5)


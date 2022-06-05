import numpy as np
import pickle
import matplotlib.pyplot as plt


def draw_convergence_plot(filename: str):
    data = pickle.load(open(filename, 'rb'))

    plt.figure()
    subplot_index = 1
    for (max_gen, pop_size) in data.keys():
        curr_run = data[(max_gen, pop_size)]
        plt.subplot(2, 3, subplot_index)
        subplot_index += 1

        plt.title(f'Instance 1; Generations = 100; Population size = {max_gen}')
        plt.xlabel('Generation')
        plt.ylabel('Makespan')

        generations = curr_run.keys()
        mins = [curr_run[gen]['min'] for gen in generations]
        avgs = [curr_run[gen]['avg'] for gen in generations]

        plt.plot(generations, avgs, label='Average makespan')
        plt.plot(generations, mins, label='Minimum makespan')
        plt.legend()

    plt.show()


def draw_multi_run_plot(filename: str):
    data = pickle.load(open(filename, 'rb'))

    plt.figure()

    for (max_gen, pop_size, index) in data.keys():
        curr_run = data[(max_gen, pop_size, index)]

        generations = curr_run.keys()
        mins = [curr_run[gen]['min'] for gen in generations]
        avgs = [curr_run[gen]['avg'] for gen in generations]

        plt.plot(generations, mins, label='Minimum makespan')
        plt.plot(generations, avgs, label='Minimum makespan')
        # plt.legend()

    plt.show()


def draw_box_plot_multirun(filename: str):
    data = pickle.load(open(filename, 'rb'))

    plt.figure()

    all_min_vectors = []
    all_avg_vectors = []

    for (max_gen, pop_size, index) in data.keys():
        curr_run = data[(max_gen, pop_size, index)]

        generations = curr_run.keys()
        mins = [curr_run[gen]['min'] for gen in generations if gen == 0 or (gen + 1) % 10 == 0]
        avgs = [curr_run[gen]['avg'] for gen in generations if gen == 0 or (gen + 1) % 10 == 0]

        all_min_vectors.append(mins)
        all_avg_vectors.append(avgs)

    plt.subplot(1, 2, 1)
    all_min_vectors = np.array(all_min_vectors)
    all_min_vectors.transpose()
    plt.boxplot(all_min_vectors, positions=np.arange(0, max_gen + 10, 10))

    plt.subplot(1, 2, 2)
    all_avg_vectors = np.array(all_avg_vectors)
    all_avg_vectors.transpose()
    plt.boxplot(all_avg_vectors, positions=np.arange(0, max_gen + 10, 10))

    plt.show()


# draw_convergence_plot('./ga_results/results_2022_05_31_19-22-07.csv')
# draw_multi_run_plot('./ga_results/multirun_results_2022_06_01_10-17-58.csv')
# draw_box_plot_multirun('./ga_results/multirun_results_2022_06_01_13-56-48.csv')
draw_box_plot_multirun('./ga_results/multirun_results_2022_06_04_15-33-14.csv')
draw_box_plot_multirun('./ga_results/multirun_results_2022_06_03_17-31-29.csv')
draw_multi_run_plot('./ga_results/multirun_results_2022_06_04_15-33-14.csv')

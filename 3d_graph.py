import numpy as np
import pickle
import matplotlib.pyplot as plt


data = pickle.load(open('res.csv', 'rb'))
# print(data)


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

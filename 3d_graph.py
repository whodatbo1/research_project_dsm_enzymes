import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = np.array(pickle.load(open('res.csv', 'rb')))

# print(data[:, 0])
# plt.figure()
# plt.plot(data[:, 2], data[:, 3])
# plt.plot(data[:, 2], data[:, 4])
# for data_point in data:
#     plt.plot(data_point[2], data_point[3])
# plt.show()
# fig = plt.figure()
# # ax = plt.axes(projection='3d')
#
# for gen in np.unique(data[:, 0]):
#     ind = np.where(data[:, 0] == gen)[0]
#     # print(data[ind])
#     plt.plot(data[ind][:, 2], data[ind][:, 3], label='Population size = ' + str(int(gen)))
#
#
# # ax.plot3D(data[:, 2], data[:, 0], data[:, 3])
# plt.legend()
# plt.title('Results for instance 1')
# plt.xlabel('Generation count')
# plt.ylabel('Minimum makespan')
# plt.show()

plt.figure()
subplot_index = 1
for gen in np.unique(data[:, 0]):
    ind = np.where(data[:, 0] == gen)[0]
    # print(data[ind])
    plt.subplot(2, 3, subplot_index)
    subplot_index += 1
    plt.title(f'Instance 1; Generations = 100; Population size = {gen}')
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.plot(data[ind][:, 2], data[ind][:, 4], label='Average makespan')
    plt.plot(data[ind][:, 2], data[ind][:, 3], label='Minimum makespan')
    plt.legend()
plt.show()

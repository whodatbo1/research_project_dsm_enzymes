from email.utils import localtime
from operator import attrgetter
import matplotlib.pyplot as plt
from numpy import average

from pandas import DataFrame
from regex import I
from instance import instance
from graphSchedule import graphSchedule

def fjsProblem(i, tabu_length, steps, steps_improved, perMachine):
    # obatin the initial schedule and makespan from the instance
    inst = instance(i)
    initial = inst.globalSelectionInitialSchedule()

    # set up values for the optimal schedule
    initialJS = jsProblem(initial, tabu_length, steps, steps_improved)
    inst.fjsResults.append(initialJS.makeSpan)
    optimal = initialJS
    optimal_value = initialJS.makeSpan

    # set up the loop with tracking values and the current schedule
    tabu = []
    current = initial
    s = 0
    s_improved = 0

    # run the loop as long as none of the stopping conditions are met
    while (s < steps and s_improved < steps_improved):
        # get a neighborhood list of schedules and their makespans
        neighborhood = current.getNeighborhoodAssignment2(perMachine)

        # find the best move that is not taboo
        if (len(neighborhood) == 0):
            return optimal, inst.fjsResults
        move = max(neighborhood, key=attrgetter("makeSpan"))
        while (move in tabu):
            neighborhood.remove(move)
            if (len(neighborhood) == 0):
                return optimal, inst.fjsResults
            move = max(neighborhood, key=attrgetter("makeSpan"))

        # run the best found schedule through the scheduling algorithm
        jsp_move = jsProblem(move, tabu_length, steps, steps_improved)

        inst.fjsResults.append(jsp_move.makeSpan)
        # put the last move in the tabu list and then make the next one
        tabu.append(current)
        if (len(tabu) > tabu_length):
            tabu.pop(0)

        # update all tracking values if necessary
        current = jsp_move
        s += 1
        s_improved += 1
        if (jsp_move.makeSpan <= optimal_value):
            if (jsp_move.makeSpan < optimal_value):
                s_improved = 0
            optimal_value = jsp_move.makeSpan
            optimal = jsp_move
    
    # return the best obtained schedule
    return optimal, inst.fjsResults
    

def jsProblem(sched: graphSchedule, tabu_length, steps, steps_improved):
    # set up values for the optimal schedule
    optimal = sched
    optimal_value = sched.makeSpan

    # set up the loop with tracking values and the current schedule
    tabu = []
    current = sched
    s = 0
    s_improved = 0

    # run the loop as long as none of the stopping conditions are met
    while (s < steps and s_improved < steps_improved):
        # get a neighborhood list of schedules and their makespans
        neighborhood = current.getNeighborhoodSequencing()

        # find the best move that is not taboo
        if (len(neighborhood) == 0):
            return optimal
        move = max(neighborhood, key=attrgetter("makeSpan"))
        while (move in tabu):
            neighborhood.remove(move)
            if (len(neighborhood) == 0):
                return optimal
            move = max(neighborhood, key=attrgetter("makeSpan"))

        # put the last move in the tabu list and then make the next one
        tabu.append(current)
        if (len(tabu) > tabu_length):
            tabu.pop(0)

        # update all tracking values if necessary        
        current = move
        s += 1
        s_improved += 1
        if (move.makeSpan <= optimal_value):
            if (move.makeSpan < optimal_value):
                s_improved = 0
            optimal_value = move.makeSpan
            optimal = move

    # return the best obtained schedule
    return optimal


def feasibleSchedule(inst: instance, df: DataFrame):
    df.sort_values(by = ["Start"], inplace = True)
    jobOps = []
    jobCompletion = []
    for i in range(inst.nrJobs):
        jobOps.append(0)
        jobCompletion.append(0)
    lastEnzyme = []
    endingTime = []
    for i in range(inst.nrMachines):
        lastEnzyme.append("")
        endingTime.append(0)

    if (len(df) != inst.opsPerJobAcc[-1]):
        return False

    infeasible = False
    for row in df.values:
        m = row[0]
        j = row[1]
        p = row[2]
        o = row[3]
        s = row[4]
        d = row[5]
        c = row[6]

        # check if operations of one job are performed in order
        if (o != jobOps[j] or jobCompletion[j] > s):
            infeasible = True

        # check if the operation is performed on a viable machine
        if (not m in inst.machineAlternatives[(j, o)]):
            infeasible = True

        # check if completion time and duration are correct
        if (s + d != c or d != inst.processingTimes[(j, o, m)]):
            infeasible = True

        # check if changeOver times are fulfilled
        changeOver = 0
        if (lastEnzyme[m] != ""):
            changeOver = inst.changeOvers[(m, lastEnzyme[m], p)]
        if (endingTime[m] + changeOver > s):
            infeasible = True
        
        # update trackers
        jobOps[j] += 1
        jobCompletion[j] = c
        lastEnzyme[m] = p
        endingTime[m] = c

        if infeasible:
            return False
    return True

# initialization graphs
"""
x = []
y = []
z = []
a = []
b = [26, 44, 55, 67, 78, 90, 101, 113, 124, 136, 147, 159, 170]
for i in range(13):
    y.append([])
    x.append(i)
for k in range(50):
    for i in range(13):
        inst = instance(i)
        initial = inst.globalSelectionInitialSchedule()
        y[i].append(initial.makeSpan)
for i in range(13):
    z.append(round(sum(y[i]) / 50))
    a.append(min(y[i]))

plt.plot(x, b, marker = "o", markersize = 8, label = "by job length")
plt.plot(x, z, marker = "o", markersize = 8, label = "random - average makespan")
plt.plot(x, a, marker = "o", markersize = 8, label = "random - minimal makespan")
plt.xlabel("Instance")
plt.ylabel("Makespan")
plt.title("Initialization")
plt.legend()
plt.show()
"""

# running graphs
"""
x = []
steps = 10

for j in range(steps + 1):
        x.append(j)

for i in range(13):

    t0 = localtime()
    r, y = fjsProblem(i, 5, steps, steps, 2)
    t1 = localtime()

    print("Final makespan of instance", i, ":", r.makeSpan)
    t = t1 - t0
    print(t)

    plt.plot(x, y, marker = "o", markersize = 8, label = "instance: " + str(i))
plt.xlabel("Instance")
plt.ylabel("Makespan")
plt.title("Makespan of chosen moves over iterations")
plt.legend()
plt.show()
"""

# running times
"""
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y5 = [0.60, 1.25, 1.71, 2.49, 3.12, 5.70, 7.45, 14.45, 18.06, 11.32, 17.15, 21.27, 31.55]
y10 = [1.71, 3.11, 6.07, 7.47, 11.82, 16.48, 22.08, 41.24, 33.51, 87.72, 97.99, 198.53, 405.67]
y20 = [3.67, 12.64, 16.58, 27.84, 32.93, 59.47, 77.27, 154.05, 161.73, 235.39, 244.55, 407.99, 1093.73]
plt.plot(x, y5, marker = "o", markersize = 8, label = "iterations: 5")
plt.plot(x, y10, marker = "o", markersize = 8, label = "iterations: 10")
plt.plot(x, y20, marker = "o", markersize = 8, label = "iterations: 20")
plt.xlabel("Instance")
plt.ylabel("Time")
plt.title("Running times")
plt.legend()
plt.show()
"""

# MILP comparisons
"""
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
i = [26, 44, 55, 67, 78, 90, 101, 113, 124, 136, 147, 159, 170]
milp1 = [22, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
milp5 = [22, 34, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
milp15 = [22, 33, 46, 66, 82, 145, 0, 0, 0, 0, 0, 0, 0]
milp30 = [22, 33, 45, 61, 82, 93, 142, 0, 0, 0, 0, 0, 0]
milp900 = [22, 33, 43, 56, 68, 93, 100, 115, 128, 144, 163, 207, 214]
milp2700 = [22, 33, 43, 55, 68, 82, 96, 114, 128, 144, 160, 180, 193]


plt.plot(x, i, marker = "o", markersize = 8, label = "Initialization (tabu search)")
plt.plot(x, milp1, marker = "o", markersize = 8, label = "MILP (time = 1s)")
plt.plot(x, milp5, marker = "o", markersize = 8, label = "MILP (time = 5s)")
plt.plot(x, milp15, marker = "o", markersize = 8, label = "MILP (time = 15s)")
plt.plot(x, milp30, marker = "o", markersize = 8, label = "MILP (time = 30s)")
plt.plot(x, milp900, marker = "o", markersize = 8, label = "MILP (time = 900s)")
plt.plot(x, milp2700, marker = "o", markersize = 8, label = "MILP (time = 2700s)")
plt.xlabel("Instance")
plt.ylabel("Makespan")
plt.title("Algorithm comparison")
plt.legend()
plt.show()
"""

# bigger instance set (needs adaptation of the instance class to get the instance information)
x = []
y = []

a = []
b = []

for i in range(13):
    temp = []
    for j in range(10):
        inst = instance(i, j)
        initial = inst.globalSelectionInitialSchedule()
        x.append(i * 10 + j)
        y.append(initial.makeSpan)
        temp.append(initial.makeSpan)
    a.append(i * 10 + 5)
    b.append(average(temp))

plt.plot(x, y, marker = "o", markersize = 6, label = "makespan")
plt.plot(a, b, marker = "o", markersize = 6, label = "average")
plt.xlabel("Instance")
plt.ylabel("Makespan")
plt.title("Initialisation on a set of random instances")
plt.legend()
plt.show()
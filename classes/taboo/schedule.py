import numpy as np
import pandas as pd
import itertools as it
from instance import instance


# two-vector representation of a schedule
# inst: the instance to which the schedule belongs to
# v1: list stating the machine on which each operation of each job is performed
# v2: list determining the order in which operations are performed
# dataFrame: dataframe representation of the current schedule
# makeSpan: the total amount of time needed to finish all jobs with the current schedule
# machineEdges: dictionary keeping track of all operations performed on the same machine
# chosenEdges: dictionary keeping track of all edges in the graph representation of the schedule
# criticalPath: list of nodes in the graph that contribute to the makespan of the schedule
class schedule:
    def __init__(self, inst: instance, v1, v2):
        self.inst = inst
        self.v1 = v1
        self.v2 = v2
        self.dataFrame = self.getDataframe()
        self.makeSpan = self.calculateMakespan()
        self.machineEdges, self.chosenEdges = self.addScheduleEdges()
        self.criticalPath = self.getCriticalPath(0, 0)

    # calculates the makespan of a schedule from a dataFrame schedule
    def calculateMakespan(self):
        comp_times = self.dataFrame["Completion"]
        return comp_times.max()

    # converts a schedule in vector representation into a schedule in dataFrame representation
    def getDataframe(self):
        time_array = np.zeros(self.inst.nrMachines) #last use time of each machine
        last_enzyme = np.zeros(self.inst.nrMachines) #last enzym used on each machine
        operations_performed = np.zeros(self.inst.nrJobs) #number of operations performed on each job
        operation_completion = np.zeros(self.inst.nrJobs) #completion time of last operation on each job
        
        results = []
        # go through each scheduled operation by looking at the number of occurrences of each job
        for j in self.v2:
            o = self.inst.operations[j][operations_performed[j]] #the current operation
            p = self.inst.orders[j]["product"] #the current product

            # find the machine on which the operation is performed
            ind = operations_performed[j] + self.inst.operationsPerJobAcc[j]
            m = self.v1[ind]

            # duration of the operation
            d = self.inst.processingTimes[(j, o, m)]

            # get the changeOver time if there is any
            changeOver = 0
            if last_enzyme[m] != 0:
                changeOver = self.inst.changeOvers[(m, last_enzyme[m], p)]

            # starting time of the operation
            s = max(operation_completion[j] + changeOver, time_array[m])

            # store the schedule entry in results
            res = {"Machine": m, "Job": j, "Product": p, "Operation": o, "Start": s, "Duration": d, "Completion": s + d}
            results.append(res)

            # update all necessary values
            time_array[m] = s + d
            last_enzyme[m] = p
            operations_performed[j] += 1
            operation_completion[j] = s + d

        dfSchedule = pd.DataFrame(results)
        dfSchedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
        dfSchedule.to_csv('csv_output.csv', index=False)
        return dfSchedule

    # creates two additional lists of edges
    # possible edges keeep track of all possible combinations of schedules for each machine
    # chosen edges (subset of possible edges) are the order in which operations are performed on each machine
    def addScheduleEdges(self):
        machineEdegs = {} # edges connecting all operations on the same machine
        chosenEdges = self.inst.edges # edges determining the order of operations on a machine

        opsOnMachine = [] # list of which operations are on which machine
        for i in range(self.inst.nrMachines):
            opsOnMachine.append([])
        opInJob = np.zeros(self.inst.nrJobs) # keeps track of the next operation to be performed on each job

        # loop over all operations and find their respective job and machine
        for j in self.v2:
            op = self.inst.operations[j][opInJob[j]]
            index = self.inst.operationsPerJobAcc[j] + op
            m = self.v1[index]

            op += 1 # add 1 to value to get the right node
            for i in range(len(opsOnMachine[m])):
                # determine the costs of the edges
                enzyme0 = self.inst.orders[j]["product"]
                enzyme1 = self.inst.orders[opsOnMachine[i][0]]["product"]
                changeOver01 = self.inst.changeOvers((m, enzyme0, enzyme1))
                changeOver10 = self.inst.changeOvers((m, enzyme1, enzyme0))

                # add all possible edges
                machineEdegs[op] += [(opsOnMachine[i][1], changeOver01)]
                machineEdegs[opsOnMachine[i][1]] += [(op, changeOver10)]

                # add the precedence constraint edges
                if (i > 0 and i == len(opsOnMachine[m] - 1)):
                    chosenEdges[opsOnMachine[i][1]] += [(op, changeOver10)]

            # update the tracking lists accordingly
            opsOnMachine[m].append((j, op))
            opInJob[j] += 1

        return machineEdegs, chosenEdges

    def getCriticalPath(self, current, cost):

        # update cost with the cost of the current node
        cost += self.inst.nodes[current]

        # if the makespan is reached, return the tail of the path
        if (cost == self.makeSpan):
            return [current, self.inst.nodes[len(self.inst.nodes) - 1]]

        # if the end node is reached without the correct cost, return nothing
        if (current == self.inst.nodes[len(self.inst.nodes) - 1]):
            return []
    
        # loop over all outgoing edges of the current node
        for next in self.chosenEdges[current]:
            # recursively build the path
            edgeCost = next[1]
            path = [current] + self.getCriticalPath(next[0], cost + edgeCost)

            # return the path if complete, nothing if not
            if (path[-1] == self.inst.nodes[len(self.inst.nodes) - 1]):
                return path
            return []
        return []


    def neigborhoodAssignment(self):
        return [self, self]

    def neighborhoodSequencing(self):
        result = []

        for node in self.criticalPath:
            orderings = it.permutations(self.machineEdges[node])

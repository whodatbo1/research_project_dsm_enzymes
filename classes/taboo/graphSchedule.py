import numpy as np
import pandas as pd
import random
import instance

class graphSchedule:
    def __init__(self, inst, machines, possibleEdges, chosenEdges):
        self.inst: instance = inst
        self.nodes: dict[int, int] = inst.nodes
        self.zeroEdges: dict[int, list[tuple[int, int]]] = inst.zeroEdges
        self.machines: dict[int, list[tuple[int, int]]] = machines
        self.possibleEdges: dict[int, list[tuple[int, int]]] = possibleEdges
        self.chosenEdges: dict[int, list[tuple[int, int]]] = chosenEdges
        self.dataFrame = self.getDataFrame()
        self.makeSpan: int = self.calculateMakespan()
        self.criticalPath: list[int] = self.getCriticalPath(0, 0)

    # equality function for two instances of graphSchedule
    def __eq__(self, o):
        if (isinstance(o, graphSchedule)):
            a = self.nodes == o.nodes
            b = self.zeroEdges == o.zeroEdges
            c = self.machines == o.machines
            d = self.possibleEdges == o.possibleEdges
            e = self.chosenEdges == o.chosenEdges
            return a and b and c and d and e
        return False

    def getNeighborhoodAssignment(self, size):
        result = []
        s = 0

        l = len(self.nodes) - 1
        incomingEdges = self.getIncomingEdges()

        for current in self.criticalPath:
            # break if the size limit of the neighborhood is reached
            if (s >= size):
                break

            # if one of the dummy nodes is reached, skip the iteration
            if (current == 0 or current == l):
                continue

            # find out which job the node belongs to and the operation in that job
            j = 0
            while (current > self.inst.opsPerJobAcc[j]):
                j += 1
            j -= 1
            o = current - self.inst.opsPerJobAcc[j] - 1

            # find the machine the operation is performed on
            m = -1
            for key, val in self.machines.items():
                if (current, j) in val:
                    m = key
                    break
                        
            # loop over all other possible machines for that operation
            possibleMachines = self.inst.machineAlternatives[(j, o)]
            possibleMachines.remove(m)
            for pickMachine in possibleMachines:
                # break if the size limit of the neighborhood is reached
                if (s >= size):
                    break

                # pick a random operation to swap with and ensure it is not on the critical path
                pick = random.choice(self.machines[pickMachine])
                while (pick[0] in self.criticalPath):
                    pick = random.choice(self.machines[pickMachine])
                nPick = pick[0]
                jPick = pick[1]

                # create new dictionary for machines and adapt according to the swap
                newMachines = self.machines.copy()
                newMachines[m] = self.machines[m].copy()
                newMachines[m].remove((current, j))
                newMachines[m].append(pick)
                newMachines[pickMachine] = self.machines[pickMachine].copy()
                newMachines[pickMachine].remove(pick)
                newMachines[pickMachine].append((current, j))

                # create new dictionaries for all edges
                newPossibleEdges = self.possibleEdges.copy()
                newChosenEdges = self.chosenEdges.copy()

                # all edges coming out of the current node will now start at the picked node
                newPossibleEdges[nPick] = []
                newChosenEdges[nPick] = []
                for t in self.possibleEdges[current]:
                    newPossibleEdges[nPick].append((t[0], self.getCorrectCost(jPick, t[0], m, True)))
                for t in self.chosenEdges[current]:
                    newChosenEdges[nPick].append((t[0], self.getCorrectCost(jPick, t[0], m, True)))

                # all edges coming out of the picked node will now start at the current node
                newPossibleEdges[current] = []
                newChosenEdges[current] = []
                for t in self.possibleEdges[nPick]:
                    newPossibleEdges[current].append((t[0], self.getCorrectCost(j, t[0], pickMachine, True)))
                for t in self.chosenEdges[nPick]:
                    newChosenEdges[current].append((t[0], self.getCorrectCost(j, t[0], pickMachine, True)))

                # all edges going into the current node will now go into the picked node
                for n in incomingEdges[current]:
                    newPossibleEdges[n] = {}
                    newChosenEdges[n] = {}
                    for t in self.possibleEdges[n]:
                        if (t[0] == current):
                            newPossibleEdges[n].append((nPick, self.getCorrectCost(jPick, n, m, False)))
                        else: newPossibleEdges[n].append(t)
                    for t in self.chosenEdges[n]:
                        if (t[0] == current):
                            newChosenEdges[n].append((nPick, self.getCorrectCost(jPick, n, m, False)))
                        else: newChosenEdges[n].append(t)

                # all edges going into the picked node will now go into the current node
                for n in incomingEdges[nPick]:
                    newPossibleEdges[n] = {}
                    newChosenEdges[n] = {}
                    for t in self.possibleEdges[n]:
                        if (t[0] == current):
                            newPossibleEdges[n].append((current, self.getCorrectCost(j, n, pickMachine, False)))
                        else: newPossibleEdges[n].append(t)
                    for t in self.chosenEdges[n]:
                        if (t[0] == current):
                            newChosenEdges[n].append((current, self.getCorrectCost(j, n, pickMachine, False)))
                        else: newChosenEdges[n].append(t)

                result.append(graphSchedule(self.inst, newMachines, newPossibleEdges, newChosenEdges))
                s += 1
        return result

    def getCorrectCost(self, j, n, m, jobToNode):
        # find the job of the obtained node
        jn = 0
        while (n > self.inst.opsPerJobAcc[jn]):
            jn += 1
        jn -= 1

        # find the enzymes belonging to each job
        enzyme = self.inst.orders[j]["product"]
        enzymen = self.inst.orders[jn]["product"]

        # jobToNode determines the way the edge goes
        if jobToNode: return self.inst.changeOvers[(m, enzyme, enzymen)]
        else: return self.inst.changeOvers[(m, enzymen, enzyme)]


    def getNeighborhoodSequencing(self, size):
        result = []
        s = 0

        l = len(self.nodes) - 1

        for i in range(len(self.criticalPath)):
            # break if the size limit of the neighborhood is reached
            if (s >= size):
                break

            # get the current and next node in the sequence
            current = self.criticalPath[i]
            if (current == l):
                continue
            next = self.criticalPath[i + 1]

            # edges with zero cost (dummy edges or job precedence constraints) cannot be changed
            if ((next, 0) in self.zeroEdges[current]):
                continue

            # create a new set of chosen edges
            newChosenEdges: dict[int, list[tuple[int, int]]] = {}
            for n in self.nodes:
                newChosenEdges[n] = []
            
            # loop over all existing edges for each node
            for n in self.nodes:
                # the edge going from current to next gets inversed
                if (n == current):
                    for t in self.chosenEdges[n]:
                        if (t[0] == next):
                            newChosenEdges[next].append((current, self.getCost(next, current)))
                        else:
                            newChosenEdges[n].append(t)

                # all edges starting at next now start at current
                elif (n == next):
                    for t in self.chosenEdges[n]:
                        newChosenEdges[current].append((t[0], self.getCost(current, t[0])))

                # all edges going into current now go into next
                else:
                    for t in self.chosenEdges[n]:
                        if (t[0] == current):
                            newChosenEdges[n].append((next, self.getCost(n, next)))
                        else:
                            newChosenEdges[n].append(t)

            # add the new schedule to the neighborhood
            result.append(graphSchedule(self.inst, self.machines, self.possibleEdges, newChosenEdges))
            s += 1
        return result

    # finds the cost of edge going from the origin node to the destination node
    def getCost(self, o, d):
        for t in self.possibleEdges[o]:
            if (t[0] == d):
                return t[1]
        return 0

    def getDataFrame(self):
        incomingEdges = self.getIncomingEdges()

        visited = [True] # keeps track if a node has been visited or not
        for i in range(len(self.nodes) - 1):
            visited.append(False)
        timeArray = np.zeros(self.inst.nrMachines) # last use time of each machine
        lastEnzyme = np.zeros(self.inst.nrMachines) # last enzyme used on each machine
        previousOpCompletion = np.zeros(self.inst.nrJobs) # completion time of last operation on each job

        l = len(self.nodes) - 1 # number of nodes (for computational simplicity)
        queue: list[int] = [] # a queue to keep track of the next nodes to visit
        for t in self.zeroEdges[0]:
            queue.append(t[0])

        result = []
        while (len(queue) != 0):
            n = queue.pop(0)

            # if the ending node is reached or not all previous nodes have been reached, skip this iteration
            if (n == l):
                continue
            skipping =  False
            for i in (incomingEdges[n]):
                if (not visited[i]):
                    skipping = True
                    break
            if skipping:
                continue

            # find out which job the node belongs to
            j = 0
            while (n > self.inst.opsPerJobAcc[j]):
                j += 1
            j -= 1

            # find the correct operation in that job
            o = n - 1 - self.inst.opsPerJobAcc[j]

            # find the product of that job
            p = self.inst.orders[j]["product"]

            # find the machine the operation is performed on
            m = -1
            for key, val in self.machines.items():
                if (n, j) in val:
                    m = key
                    break

            # find the duration of the operation
            d = self.inst.processingTimes[(j, o, m)]

            # find the starting time of the operation
            changeOver = 0
            if (lastEnzyme[m] != 0):
                changeOver = self.inst.changeOvers[(m, lastEnzyme[m], p)]
            s = max(previousOpCompletion[j], timeArray[m] + changeOver)

            # store the schedule entry in results
            res = {"Machine": m, "Job": j, "Product": p, "Operation": o, "Start": s, "Duration": d, "Completion": s + d}
            result.append(res)

            # update all necessary values
            timeArray[m] = s + d
            lastEnzyme[m] = p
            previousOpCompletion[j] = s + d
            visited[n] = True
            # add all connected nodes to the queue
            for t in (self.zeroEdges[n]):
                queue.append(t[0])
            for t in (self.chosenEdges[n]):
                queue.append(t[0])


        dfSchedule = pd.DataFrame(result)
        dfSchedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
        dfSchedule.to_csv('csv_output.csv', index=False)
        return dfSchedule

    def calculateMakespan(self):
        comp_times = self.dataFrame["Completion"]
        return comp_times.max()

    def getCriticalPath(self, current, cost):
        # update cost with the cost of the current node
        cost += self.inst.nodes[current]
        last = len(self.inst.nodes) - 1

        # if the makespan is reached, return the tail of the path
        if (cost == self.makeSpan):
            return [current, self.inst.nodes[last]]

        # if the end node is reached without the correct cost, return nothing
        if (current == self.inst.nodes[last]):
            return []
    
        outgoingEdges = self.chosenEdges[current] + self.zeroEdges[current]
        # loop over all outgoing edges of the current node
        for next in outgoingEdges:
            # recursively build the path
            edgeCost = next[1]
            path = [current] + self.getCriticalPath(next[0], cost + edgeCost)

            # return the path if complete, nothing if not
            if (path[-1] == self.inst.nodes[last]):
                return path
            return []
        return []

    # returns a dictionary storing all incoming edges in the final schedule graph
    def getIncomingEdges(self):
        # initialize the dictionary
        incomingEdges: dict[int, list[int]] = {}
        for n in self.nodes:
            incomingEdges[n] = []

        # for each node, check the outgoing edges and add them to the resective entry in the dictionary
        for n in self.nodes:
            outgoingEdges = self.zeroEdges[n] + self.chosenEdges[n]
            for t in outgoingEdges:
                incomingEdges[t[0]].append(n)
        return incomingEdges
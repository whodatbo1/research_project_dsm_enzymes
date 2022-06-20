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
            a = self.machines == o.machines
            b = self.chosenEdges == o.chosenEdges
            return a and b
        return False

    def getNeighborhoodAssignment(self, size):
        result = []
        s = 0

        # stores incoming edges for each node from including possibleEdges
        incoming2: dict[int, list[int]] = {}
        for n in self.nodes:
            incoming2[n] = []
        for n in self.nodes:
            outgoingEdges = self.possibleEdges[n]
            for t in outgoingEdges:
                incoming2[t[0]].append(n)
        l = len(self.nodes) - 1

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
            possibleMachines = self.inst.machineAlternatives[(j, o)].copy()
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
                for n in incoming2[current]:
                    newPossibleEdges[n] = []
                    newChosenEdges[n] = []
                    for t in self.possibleEdges[n]:
                        if (t[0] == current):
                            newPossibleEdges[n].append((nPick, self.getCorrectCost(jPick, n, m, False)))
                        else: newPossibleEdges[n].append(t)
                    for t in self.chosenEdges[n]:
                        if (t[0] == current):
                            newChosenEdges[n].append((nPick, self.getCorrectCost(jPick, n, m, False)))
                        else: newChosenEdges[n].append(t)

                # all edges going into the picked node will now go into the current node
                for n in incoming2[nPick]:
                    newPossibleEdges[n] = []
                    newChosenEdges[n] = []
                    for t in self.possibleEdges[n]:
                        if (t[0] == nPick):
                            newPossibleEdges[n].append((current, self.getCorrectCost(j, n, pickMachine, False)))
                        else: newPossibleEdges[n].append(t)
                    for t in self.chosenEdges[n]:
                        if (t[0] == nPick):
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


    def getNeighborhoodAssignment2(self, perMachine):
        incomingPossible: dict[int, list[int]] = {}
        for n in self.nodes:
            incomingPossible[n] = []
        for n in self.nodes:
            outgoingEdges = self.possibleEdges[n]
            for t in outgoingEdges:
                incomingPossible[t[0]].append(n)

        result = []
        # loop over each machine and get the operations performed on it
        for m0 in self.inst.machines:
            ops0 = self.machines[m0]
            costs0 = self.getCosts(ops0, m0)

            for _ in range(perMachine):
                # pick the node with largest cost
                if (len(costs0) == 0):
                    break
                n0 = max(costs0, key = costs0.get)
                costs0.pop(n0)

                # get the other machines on which this job can run
                ms = []
                for v in self.inst.machineAlternatives.values():
                    if m0 in v:
                        ms = v.copy()
                        break
                ms.remove(m0)

                # loop over all of the other machines
                for m1 in ms:
                    pick = random.choice(self.machines[m1])
                    n1 = pick[0]
                    j1 = pick[1]

                    # create new dictionary for machines
                    j0 = -1
                    newMachines = self.machines.copy()
                    newMachines[m0] = self.machines[m0].copy()
                    for t in newMachines[m0]:
                        if (t[0] == n0):
                            j0 = t[1]
                            newMachines[m0].remove(t)
                            break
                    newMachines[m0].append(pick)
                    newMachines[m1] = self.machines[m1].copy()
                    newMachines[m1].remove(pick)
                    newMachines[m1].append((n0, j0))

                    newPossibleEdges = self.possibleEdges.copy()
                    newChosenEdges = self.chosenEdges.copy()
                    
                    # all edges from n0 now start at n1
                    newPossibleEdges[n1] = []
                    newChosenEdges[n1] = []
                    for t in self.possibleEdges[n0]:
                        newPossibleEdges[n1].append((t[0], self.getChangeOver(n1, t[0], m0)))
                    for t in self.chosenEdges[n0]:
                        newChosenEdges[n1].append((t[0], self.getChangeOver(n1, t[0], m0)))

                    # all edges from n1 now start at n0
                    newPossibleEdges[n0] = []
                    newChosenEdges[n0] = []
                    for t in self.possibleEdges[n1]:
                        newPossibleEdges[n0].append((t[0], self.getChangeOver(n0, t[0], m1)))
                    for t in self.chosenEdges[n1]:
                        newChosenEdges[n0].append((t[0], self.getChangeOver(n0, t[0], m1)))

                    # all edges going into n0 will now go into n1
                    for n in incomingPossible[n0]:
                        newPossibleEdges[n] = []
                        newChosenEdges[n] = []
                        for t in self.possibleEdges[n]:
                            if (t[0] == n0):
                                newPossibleEdges[n].append((n1, self.getChangeOver(n, n1, m0)))
                            else: newPossibleEdges[n].append(t)
                        for t in self.chosenEdges[n]:
                            if (t[0] == n0):
                                newChosenEdges[n].append((n1, self.getChangeOver(n, n1, m0)))
                            else: newChosenEdges[n].append(t)

                    # all edges going into n1 will now go into n0
                    for n in incomingPossible[n1]:
                        newPossibleEdges[n] = []
                        newChosenEdges[n] = []
                        for t in self.possibleEdges[n]:
                            if (t[0] == n1):
                                newPossibleEdges[n].append((n0, self.getChangeOver(n, n0, m1)))
                            else: newPossibleEdges[n].append(t)
                        for t in self.chosenEdges[n]:
                            if (t[0] == n1):
                                newChosenEdges[n].append((n0, self.getChangeOver(n, n0, m1)))
                            else: newChosenEdges[n].append(t)

                    result.append(graphSchedule(self.inst, newMachines, newPossibleEdges, newChosenEdges))
        return result

    # get the cost for each operation (duration + changeOver times) of a certain machine
    def getCosts(self, ops, m):
        costs = {}
        for t in ops:
                costs[t[0]] = 0
        for t in ops:
            o = t[0] - self.inst.opsPerJobAcc[t[1]] - 1
            costs[t[0]] += self.inst.processingTimes[(t[1], o, m)]
            for e in self.chosenEdges[t[0]]:
                costs[t[0]] += e[1]
                costs[e[0]] += e[1]
        return costs

    def getChangeOver(self, n0, n1, m):
        # find the jobs belonging to each node
        j0 = 0
        j1 = 0
        while (n0 > self.inst.opsPerJobAcc[j0]):
            j0 += 1
        while (n1 > self.inst.opsPerJobAcc[j1]):
            j1 += 1
        j0 -= 1
        j1 -= 1

        # get the enzymes
        enzyme0 = self.inst.orders[j0]["product"]
        enzyme1 = self.inst.orders[j1]["product"]
        
        # return the correct changeOver time
        return self.inst.changeOvers[(m, enzyme0, enzyme1)]

    def getNeighborhoodSequencing(self):
        result = []

        l = len(self.nodes) - 1
        for i in range(len(self.criticalPath)):

            # get the current and next node in the sequence
            current = self.criticalPath[i]
            if (current == 0 or current == l):
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
        return result

    # finds the cost of edge going from the origin node to the destination node
    def getCost(self, o, d):
        for t in self.possibleEdges[o]:
            if (t[0] == d):
                return t[1]
        return 0

    def getDataFrame(self):
        visited = [True] # keeps track if a node has been visited or not
        for i in range(len(self.nodes) - 1):
            visited.append(False)

        incomingEdges = self.getIncomingEdges()

        timeArray = [] # last use time of each machine
        lastEnzyme = [] # last enzyme used on each machine
        for i in range(self.inst.nrMachines):
            timeArray.append(0)
            lastEnzyme.append("")

        previousOpCompletion = [] # completion time of last operation on each job
        for i in range(self.inst.nrJobs):
            previousOpCompletion.append(0)

        l = len(self.nodes) - 1 # number of nodes (for computational simplicity)
        queue: list[int] = [] # a queue to keep track of the next nodes to visit
        for t in self.zeroEdges[0]:
            queue.append(t[0])

        result = []
        while (len(queue) != 0):
            n = queue.pop(0)

            # if the ending node is reached or not all previous nodes have been reached, skip this iteration
            if (n == l or visited[n]):
                continue
            skipping = False
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
            if (lastEnzyme[m] != ""):
                changeOver = self.inst.changeOvers[(m, lastEnzyme[m], p)]
            s = max(previousOpCompletion[j], timeArray[m] + changeOver)

            # store the schedule entry in results
            res = {"Machine": m, "Job": j, "Product": p, "Operation": o, "Start": s, "Duration": d, "Completion": s + d}
            result.append(res)

            # update all necessary tracking values and the queue
            timeArray[m] = s + d
            lastEnzyme[m] = p
            previousOpCompletion[j] = s + d
            visited[n] = True
            for t in (self.zeroEdges[n]):
                queue.append(t[0])
            for t in (self.chosenEdges[n]):
                queue.append(t[0])

        # convert and return the result to a dataframe
        dfSchedule = pd.DataFrame(result)
        dfSchedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
        dfSchedule.to_csv('csv_output.csv', index=False)
        return dfSchedule

    def calculateMakespan(self):
        comp_times = self.dataFrame["Completion"]
        return comp_times.max()

    def getCriticalPath(self, current, cost):
        # if the end node is reached without the correct cost, return nothing
        last = len(self.nodes) - 1
        if (current == last):
            return []

        # if the makespan is reached, return the tail of the path
        cost += self.nodes[current]
        if (cost == self.makeSpan):
            return [current, last]
    
        outgoingEdges = self.zeroEdges[current] + self.chosenEdges[current]
        # loop over all outgoing edges of the current node
        for next in outgoingEdges:
            path = [current] + self.getCriticalPath(next[0], cost + next[1])

            # return the path if complete, nothing if not
            if (path[-1] == last):
                return path
        return path

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
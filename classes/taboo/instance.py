import importlib.util
from graphSchedule import graphSchedule
import random

class instance:
    def __init__(self, i, j):
        mod = self.getInstanceInfo(i, j)
        
        self.instance = i
        self.orders: dict[int, dict] = mod.orders
        self.machines: list[int] = mod.machines
        self.nrMachines: int = mod.nr_machines
        self.jobs: list[int] = mod.jobs
        self.nrJobs: int = mod.nr_jobs
        self.operations: dict[int, list[int]] = mod.operations
        self.machineAlternatives: dict[tuple[int, int], list[int]] = mod.machineAlternatives
        self.processingTimes: dict[tuple[int, int, int], int] = mod.processingTimes
        self.changeOvers: dict[tuple[int, str, str], int] = mod.changeOvers
        self.nodes, self.zeroEdges = self.createBaseGraph()
        self.opsPerJobAcc = self.getOpsPerJobAcc()
        self.fjsResults = []

    # gets the information from an instance file
    def getInstanceInfo(self, i, j):
        fileName = 'FJSP_' + str(i) + '-' + str(j)
        spec = importlib.util.spec_from_file_location('instance', "./instances/new_instances/" + fileName + '.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # creates an initial schedule based on a global selection heuristic for the FJSP instance we are working with
    # global selection keeps track of the overall running times of each machine and assigns operations accordingly
    # the original heuristic method makes use of randomness when choosing jobs or assiging operations
    # I try to avoid this as to make the results more reproducible
    def globalSelectionInitialSchedule(self):
        # initialize empty schedule
        machines: dict[int, list[tuple[int, int]]] = {}
        for i in self.machines:
            machines[i] = []
        possibleEdges: dict[int, list[tuple[int, int]]] = {}
        chosenEdges: dict[int, list[tuple[int, int]]] = {}
        for i in self.nodes:
            possibleEdges[i] = []
            chosenEdges[i] = []

        timeArray = [] # keeps track of all machines and the workin times
        lastProduct = [] # keeps track of the last enzyme operated on on each machine
        for i in range(self.nrMachines):
            timeArray.append(0)
            lastProduct.append("")

        # loop over every available job sorted by overall cost
        jobs = self.createJobOrdering()
        js = random.sample(self.jobs, len(self.jobs))
        for j in jobs:
            enzyme = self.orders[j]["product"]
            opCompletion = 0 # keeps track of the completion time of the previous operation

            #loop over every operation of the chosen job (in order)
            for o in self.operations[j]:
                alternatives = self.machineAlternatives[(j, o)] # get all machines on which the operation can be run

                # for each machine, add the total time depenedent on the processing time of the operation
                # and the changeover with the previous operation
                # will be None if the operation cannot be performed on that machine
                tempArray = []
                for machine in range(self.nrMachines):
                    if machine in alternatives:
                        duration = self.processingTimes[(j, o, machine)]
                    
                        # if this is the first operation on a certain machine, there will be no changeOver
                        changeOver = 0
                        if (lastProduct[machine] != ""):
                            changeOver = self.changeOvers[(machine, lastProduct[machine], enzyme)]

                        val = max(timeArray[machine] + changeOver, opCompletion) + duration
                        tempArray.append(val)
                    else:
                        tempArray.append(None)

                # find the smallest time in the temporary array and corresponding machine i
                smallest = min(x for x in tempArray if x is not None)
                m = tempArray.index(smallest)

                # update tracking arrays
                timeArray[m] = smallest
                lastProduct[m] = enzyme
                opCompletion = smallest

                # find the node of the operation
                n = self.opsPerJobAcc[j] + o + 1

                # add the correct edges to the graph representation
                l = len(machines[m])
                for i in range(l):
                    t = machines[m][i]
                    enzymeT = self.orders[t[1]]["product"]

                    cost0T = self.changeOvers[(m, enzyme, enzymeT)]
                    costT0 = self.changeOvers[(m, enzymeT, enzyme)]

                    possibleEdges[n].append((t[0], cost0T))
                    possibleEdges[t[0]].append((n, costT0))

                    if (i == l - 1):
                        chosenEdges[t[0]].append((n, costT0))
                machines[m].append((n, j))

        return(graphSchedule(self, machines, possibleEdges, chosenEdges))

    # order the jobs based on the total amount of time necessary to complete them
    def createJobOrdering(self):
        times = []
        for j in self.jobs:
            times.append((0, j))
            for o in self.operations[j]:
                machines = self.machineAlternatives[(j, o)]
                times[j] = (times[j][0] + self.processingTimes[(j, o, machines[0])], times[j][1])
        result = []
        for t in sorted(times, reverse = True):
            result.append(t[1])
        return result

    # creates a basic version of a disjunctive graph representation
    def createBaseGraph(self):
        # initialize nodes and edges and add start node
        nodes = {0: 0}
        zeroEdges = {0: []}

        index = 1
        for j in self.jobs:
            for op in self.operations[j]:
                # add all the nodes corresponding to the operations
                machines = self.machineAlternatives[(j, op)]
                cost = self.processingTimes[(j, op, machines[0])]
                nodes[index] = cost
                zeroEdges[index] = []

                # add an edge from the origin node to the first operation of each job
                if (op == 0):
                    zeroEdges[0].append((index, 0))
                # add an edge from the previous operation of the same job to this node (precedence constraints)
                else:
                    zeroEdges[index - 1].append((index, 0))
                # update the index value
                index += 1

        # add the ending node
        nodes[index] = 0
        zeroEdges[index] = []
        # add an edge from the last operation of each job to the output node
        ind = 0
        for j in self.jobs:
            ind += len(self.operations[j])
            zeroEdges[ind].append((index, 0))
        return nodes, zeroEdges

    # creates a list to easily find which job an operation node belongs to
    def getOpsPerJobAcc(self):
        result = [0] 
        for j in range(self.nrJobs):
            opsInJob = len(self.operations[self.jobs[j]])
            result.append(result[j] + opsInJob)
        return result
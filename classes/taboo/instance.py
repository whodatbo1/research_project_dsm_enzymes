import importlib.util
import numpy as np
import graphSchedule

class instance:
    def __init__(self, i):
        mod = self.getInstanceInfo(i)
        
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

    # gets the information from an instance file
    def getInstanceInfo(self, i):
        fileName = 'FJSP_' + str(i)
        spec = importlib.util.spec_from_file_location('instance', "../../instances/" + fileName + '.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    # creates an initial schedule based on a global selection heuristic for the FJSP instance we are working with
    # global selection keeps track of the overall running times of each machine and assigns operations accordingly
    # the original heuristic method makes use of randomness when choosing jobs or assiging operations
    # I try to avoid this as to make the results more reproducible
    def globalSelectionInitialSchedule(self):
        # initialize empty schedule
        v1 = []
        v2 = []

        time_array = np.zeros(self.nrMachines) # keeps track of all machines and the workin times
        last_enzyme = np.zeros(self.nrMachines) # keeps track of the last enzyme operated on on each machine

        jobs = self.createJobOrdering()
        # loop over every available job sorted by overall cost
        for j in jobs:
            product = self.orders[j]["product"]
            previous_op = 0 # keeps track of the completion time of the previous operation

            #loop over every operation of the chosen job (in order)
            for op in self.operations[j]:
                alternatives = self.machineAlternatives[(j, op)] # get all machines on which the operation can be run

                # for each machine, add the total time depenedent on the processing time of the operation
                # and the changeover with the previous operation
                # will be None if the operation cannot be performed on that machine
                temp_array = []
                for machine in range(self.nr_machines):
                    if machine in alternatives:
                        duration = self.processingTimes[(j, op, machine)]
                    
                        # if this is the first operation on a certain machine, there will be no changeOver
                        changeOver = 0
                        if (last_enzyme[machine] != 0):
                            changeOver = self.changeOvers[(machine, last_enzyme[machine], product)]

                        val = max(time_array[machine] + changeOver, previous_op) + duration
                        temp_array.append(val)
                    else:
                        temp_array.append(None)

                # find the smallest time in the temporary array and corresponding machine i
                smallest = min(x for x in temp_array if x is not None)
                m = temp_array.index(smallest)

                # update the time_array, the last enzyme on that machine and the ending time of the operation
                time_array[m] = smallest
                last_enzyme[m] = product
                previous_op = smallest

                # update the schedule vector representation
                v1.append(m)
                v2.append(j)

        machines, possibleEdges, chosenEdges = self.convertToGraph(v1, v2)                
        return(graphSchedule(self, machines, possibleEdges, chosenEdges))

    # order the jobs based on the total amount of time necessary to complete them
    def createJobOrdering(self):
        times = zip(np.zeros(self.nrJobs), self.jobs)
        for j in self.jobs:
            for o in self.operations[j]:
                machines = self.machineAlternatives[(j, o)]
                times[j][0] += self.processingTimes[(j, o, machines[0])]
        result = []
        for t in sorted(times):
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
        
    # converts the initial schedule from a two vector representation to a completed graph representation
    def convertToGraph(self, v1, v2):
        opsOnMachine: dict[int, list[tuple[int, int]]] = {} # keep track of which operation is performed on which machine and its job
        possibleEdges: dict[int, list[tuple[int, int]]] = {} # edges connecting all operations on the same machine
        chosenEdges: dict[int, list[tuple[int, int]]] = {} # edges determining the order of operations on a machine
        for m in self.machines:
            opsOnMachine[m] = []
        for n in self.nodes:
            possibleEdges[n] = []
            chosenEdges[n] = []

        opInJob = np.zeros(self.nrJobs) # keeps track of the next operation to be performed on each job

        # loop over all operations and find their respective job and machine
        for j in v2:
            o = self.operations[j][opInJob[j]]
            n = self.opsPerJobAcc[j] + o + 1
            m = v1[n - 1]
            
            # loop over all other operations performed on that machine
            for i in range(len(opsOnMachine[m])):
                t = opsOnMachine[m][i]

                # determine the costs of the edges
                enzyme0 = self.orders[j]["product"]
                enzyme1 = self.orders[t[1]]["product"]
                changeOver01 = self.changeOvers[(m, enzyme0, enzyme1)]
                changeOver10 = self.changeOvers[(m, enzyme1, enzyme0)]

                # add all possible edges
                possibleEdges[n].append((t[0], changeOver01))
                possibleEdges[t[0]].append((n, changeOver10))

                # add the chosen edge into the schedule
                if (i == len(opsOnMachine[m] - 1)):
                    chosenEdges[t[0]].append((n, changeOver10))

            # update the machine lists and trackers
            opsOnMachine[m].append((n, j))
            opInJob[j] += 1

        return opsOnMachine, possibleEdges, chosenEdges

    # creates a list to easily find which job an operation node belongs to
    def getOpsPerJobAcc(self):
        result = [0] 
        for j in range(self.nrJobs):
            opsInJob = len(self.operations[self.jobs(j)])
            result.append(result[j] + opsInJob)
        return result
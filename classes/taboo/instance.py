import importlib.util
from itertools import accumulate
import numpy as np
from schedule import schedule

class instance:
    def __init__(self, i):
        mod = self.get_instance_info(i)
        
        self.instance = i
        self.orders = mod.orders
        self.machines = mod.machines
        self.nrMachines = len(self.machines)
        self.jobs = mod.jobs
        self.nrJobs = len(self.jobs)
        self.operations = mod.operations
        self.machineAlternatives = mod.machineAlternatives
        self.processingTimes = mod.processingTimes
        self.changeOvers = mod.changeOvers
        self.operationsPerJobAcc = self.getOpsPerJobAcc()
        self.nodes, self.edges = self.createGraph()

    def get_instance_info(i):
        fileName = 'FJSP_' + str(i)
        spec = importlib.util.spec_from_file_location('instance', "../../instances/" + fileName + '.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    #this method creates an initial schedule based on a global selection heuristic for the FJSP instance we are working with
    #global selection keeps track of the overall running times of each machine and assigns operations accordingly
    #the original heuristic method makes use of randomness when choosing jobs or assiging operations
    #I try to avoid this as to make the results more reproducible
    def global_selection_initial_schedule(self):
        #initialize empty schedule
        v1 = []
        v2 = []

        time_array = np.zeros(self.nrMachines) #keeps track of all machines and the workin times
        last_enzyme = np.zeros(self.nrMachines) #keeps track of the last enzyme operated on on each machine

        # loop over every available job (can be random and changed later on)
        for job in self.jobs:
            order = self.orders[job]
            ops = self.operations[job]
            previous_op = 0 #keeps track of the completion time of the previous operation

            #loop over every operation of the chosen job (in order)
            for op in ops:
                alternatives = self.machineAlternatives[(job, op)] #get all machines on which the operation can be run
                temp_array = []

                # for each machine, add the total time depenedent on the processing time of the operation
                # and the changeover with the previous operation
                # will be None if the operation cannot be performed on that machine
                for machine in range(self.nr_machines):
                    if machine in alternatives:
                        duration = self.processingTimes[(job, op, machine)]
                    
                        # if this is the first operation on a certain machine, there will be no changeOver
                        changeOver = 0
                        if last_enzyme[machine] != 0:
                            changeOver = self.changeOvers[(machine, last_enzyme[machine], order["product"])]

                        val = max(time_array[machine] + changeOver, previous_op) + duration
                        temp_array.append(val)

                    else:
                        temp_array.append(None)

                # find the smallest time in the temporary array and corresponding machine i
                smallest = min(x for x in temp_array if x is not None)
                i = temp_array.index(smallest)

                # update the time_array, the last enzyme on that machine and the ending time of the operation
                time_array[i] = smallest
                last_enzyme[i] = order["product"]
                previous_op = smallest

                # update the schedule vector representation
                v1.append(i)
                v2.append(job)
                
        return(schedule(self, v1, v2))

    # creates a list that accumulatively counts all operations per job
    # makes it easier to access the correct operation when working with graph representation or the schedule
    def getOpsPerJobAcc(self):
        accumul = [0]
        for j in self.jobs:
            accumul.append(len(self.operations[j]) + accumul[j])

    # creates a basic version of a disjunctive graph representation
    def createGraph(self):
        # initialize nodes and edges and add start node
        nodes = {0: 0}
        edges = {0: []}

        index = 1
        for j in self.jobs:
            for op in self.operations[j]:
                # add all the nodes corresponding to the operations
                machines = self.machineAlternatives[(j, op)]
                cost = self.processingTimes[(j, op, machines[0])]
                nodes[index] = cost
                edges[index] = []

                # add an edge from the origin node to the first operation of each job
                if (op == 0):
                    edges[0] += [(index, 0)]
                # add an edge from the previous operation of the same job to this node (precedence constraints)
                else:
                    edges[index - 1] += [(index, 0)]
                # update the index value
                index += 1

        # add end node
        l = len(nodes)
        nodes[l] = 0
        edges[l] = []
        # add an edge from the last operation of each job to the output node
        for j in self.jobs:
            ind = self.operationsPerJobAcc[j + 1]
            edges[ind] += [(l, 0)]
        return nodes, edges
        
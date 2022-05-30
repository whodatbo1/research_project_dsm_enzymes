import importlib

import random
import numpy as np
import pandas as pd

from classes.milp import FlexibleJobShop


def create_graph(alg, v1, v2, v3):
    jobs = alg.jobs
    # create list of vertices containing of all jobs, and the 2 dummy nodes
    vertices = {-2: 0, -1: 0}
    # create a dict with key as vertex id and the value as weight
    for j in range(len(v3)):
        vertices.update({
            j: 0,
        })
    # Create way te represent edges of the graph, take into account the precedence arcs, machine arcs and dummy arcs.
    dummy_edges = {}
    pre_edges = {}
    machine_edges = {}
    for v in vertices.keys():
        dummy_edges.update({v: {}})
        pre_edges.update({v: {}})
        machine_edges.update({v: {}})
    dummies = {}

    for i in range(len(v3)):
        # To create the dummy edges
        if v3[i] == 0:
            dummies.update({i: 0})
            if i - 1 > 0:
                dummy_edges.update({i - 1: {-1: 0}})
        # statement for the precedence job edges
        if i + 1 < len(v3) and v3[i + 1] != 0:
            edges = {i + 1: 0}
            pre_edges.update({i: edges})
    dummy_edges.update({-2: dummies})
    # lastly fill the graph with machine relations
    return Graph(vertices, {}, dummy_edges, pre_edges, {})


def create_machine_edges(alg, v1, v3, g):
    machine_edges = {}
    for v in g.vertices.keys():
        machine_edges.update({v: {}})
    machines = {}
    for m in alg.machines:
        machines.update({m: []})
    # create a dictionary where all operations that make use of a machine are listed
    for i in range(len(v1)):
        operations = machines.get(v1[i])
        operations.append(i)
        machines.update({v1[i]: operations})
    prev_job = -1
    for m in range(len(machines)):
        ops = machines.get(m)

        for i in range(len(ops)):
            j_o = get_job_op(v3, ops[i])
            weight = alg.processingTimes[j_o, v3[ops[i]], m]
            g.vertices.update({ops[i]: weight})
            if i + 1 < len(ops):
                j_o_n = get_job_op(v3, ops[i + 1])
                co = alg.changeOvers[m, alg.orders[j_o].get("product"), alg.orders[j_o_n].get("product")]
                machine_edges.update({ops[i]: {ops[i + 1]: co}})
    return Graph(g.vertices, machines, g.dummy_edges, g.pre_edges, machine_edges)


def k_insertion(alg, g, v3, v, pos):
    m_e = g.machine_edges.copy()
    m_e.update({v: {}})

    # Assign new machine, determine order and set weight
    machines = g.machines.copy()
    for k in machines.keys():
        k_list = m_e.get(k)
        k_keys = k_list.keys()
        if v in k_keys:
            m_e.get(k).pop(v)
            break
    for k in machines:
        if v in machines[k]:
            machines[k].remove(v)
    g_min = Graph(g.vertices.copy(), machines.copy(), g.dummy_edges.copy(), g.pre_edges.copy(), m_e.copy())
    j = get_job_op(v3, v)
    o = v3[v]
    m_a = alg.machineAlternatives[j, o].copy()
    # Figure out how to determine m
    m = m_a[pos]
    qk = machines.get(m).copy()
    rk = []
    lk = []
    for x in qk:
        cm = get_critical_path(g)
        s_v = get_e_start_time(g_min, v)
        t_v = get_l_completion_time(g_min, cm[0], v)
        j_x = get_job_op(v3, x)
        s_x = get_e_start_time(g_min, x)
        t_x = get_l_completion_time(g_min, cm[0], x)
        p_x = alg.processingTimes[j_x, v3[x], m]
        if s_x + p_x > s_v:
            rk.append(x)
        if p_x + t_x > t_v:
            lk.append(x)
    rk_lk = []
    for x in qk:
        if x in lk and x in rk:
            rk_lk.append(x)
    if len(rk_lk) > 0:
        index = random.randrange(0, len(rk_lk)+1)
        qk.insert(index, v)
    elif len(rk_lk) == 0:
        q_k = qk.copy()
        for x in qk:
            if x in lk or x in rk:
                q_k.remove(x)
        if len(q_k) > 0:
            index = random.randrange(0, len(q_k) + 1)
            qk.insert(index, v)
        else:
            qk.insert(len(lk)-1, v)
    machines.update({m: qk})
    for i in range(len(qk)):
        j = get_job_op(v3, qk[i])
        if i + 1 < len(qk):
            j_o_n = get_job_op(v3, qk[i + 1])
            co = alg.changeOvers[m, alg.orders[j].get("product"), alg.orders[j_o_n].get("product")]
            m_e.update({qk[i]: {qk[i + 1]: co}})
    g = Graph(g.vertices.copy(), machines.copy(), g.dummy_edges.copy(), g.pre_edges.copy(), m_e.copy())
    return g


def get_e_start_time(g, v):
    m_p = -1
    j_p = -1
    for x in g.vertices.keys():
        if v in g.machine_edges.get(x).keys():
            m_p = x
        if v in g.pre_edges.get(x).keys():
            j_p = x
    if m_p == -1 and j_p == -1:
        return 0
    else:
        s_m = 0
        if m_p > -1:
            s_m = g.vertices.get(m_p) + g.machine_edges.get(m_p).get(v) + get_e_start_time(g, m_p)
        s_j = 0
        if j_p > -1:
            s_j = g.vertices.get(j_p) + g.pre_edges.get(j_p).get(v) + get_e_start_time(g, j_p)
        return max(s_m, s_j)


def get_l_completion_time(g, cm, v):
    m_s = -1
    if len(g.machine_edges.get(v).keys()) != 0:
        m_s = list(g.machine_edges.get(v).keys())[0]
    j_s = -1
    if len(g.pre_edges.get(v).keys()) != 0:
        j_s = list(g.pre_edges.get(v).keys())[0]
    if j_s == -1 and m_s == -1:
        return cm
    else:
        s_m = 10000
        if m_s > -1:
            keys = list(g.machine_edges.get(v).keys())
            s_m = get_l_completion_time(g, cm, m_s) - g.vertices.get(m_s) - g.machine_edges.get(v).get(keys[0])
        s_j = 10000
        if j_s > -1:
            keys = list(g.pre_edges.get(v).keys())
            s_j = get_l_completion_time(g, cm, j_s) - g.vertices.get(j_s) - g.pre_edges.get(v).get(keys[0])
        return min(s_m, s_j)


def get_job_op(v3, v):
    j = -1
    for x in range(len(v3)):
        if v3[x] == 0:
            j += 1
        if v == x:
            break
    return j


def get_critical_path(g):
    edges = {}
    for k in g.vertices.keys():
        e = g.dummy_edges.get(k)
        e.update(g.pre_edges.get(k))
        e.update(g.machine_edges.get(k))
        edges.update({k: e})
    visited = {}
    for i in g.vertices.keys():
        visited.update({i: False})
    path = []
    mkspns = []
    paths = []
    get_path(g, edges, -2, -1, visited, path, paths, 0, mkspns)
    max_m = max(mkspns)
    index = mkspns.index(max_m)
    c_p = paths[index]
    return max_m, c_p


def get_path(g, edges, c, t, visited, path, paths, m, max_m):
    visited.update({c: True})
    path.append(c)
    m += g.vertices.get(c)
    if c == t:
        max_m.append(m)
        paths.append(path.copy())
    else:
        for i in edges.get(c):
            test = edges.get(c)
            if not visited.get(i):
                co = test.get(i)
                m += co
                get_path(g, edges, i, t, visited, path, paths, m, max_m)

    path.pop()
    visited.update({c: False})


def get_data_frame(self, inst):
    incomingEdges = get_incoming_edges(self)

    visited = [True]  # keeps track if a node has been visited or not
    for i in range(len(self.vertices) - 1):
        visited.append(False)
    timeArray = np.zeros(len(inst.machines))  # last use time of each machine
    lastEnzyme = np.zeros(len(inst.machines))  # last enzyme used on each machine
    previousOpCompletion = np.zeros(len(inst.jobs))  # completion time of last operation on each job

    opsPerJobAcc = [0]  # list to easily find which job an operation node belongs to
    for j in range(len(inst.jobs)):
        opsInJob = inst.operations[inst.jobs[j]]

        opsPerJobAcc.append(opsPerJobAcc[j] + opsInJob)

    results = []

    queue = []  # a queue to keep track of the next node to visit
    for t in self.zeroEdges[0]:
        queue.append(t[0])

    while len(queue) != 0:
        n = queue.pop(0)

        skipping = False
        for i in (incomingEdges[n]):
            if not visited[i]:
                skipping = True
        if skipping:
            continue

        # find out which job the node belongs to
        j = 0
        while n > opsPerJobAcc[j]:
            j += 1
        j -= 1

        # find the correct operation in that job
        o = n - 1 - opsPerJobAcc[j]

        # find the product of that job
        p = inst.orders[j]["product"]

        # find the machine the operation is performed on
        m = -1
        for key, val in self.machines.items():
            if n in val:
                m = key
                break

        # find the duration of the operation
        d = inst.processingTimes[(j, o, m)]

        # find the starting time of the operation
        changeOver = 0
        if lastEnzyme[m] != 0:
            changeOver = self.inst.changeOvers[(m, lastEnzyme[m], p)]
        s = max(previousOpCompletion[j], timeArray[m] + changeOver)

        # store the schedule entry in results
        res = {"Machine": m, "Job": j, "Product": p, "Operation": o, "Start": s, "Duration": d, "Completion": s + d}
        results.append(res)

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

    dfSchedule = pd.DataFrame(results)
    dfSchedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    dfSchedule.to_csv('csv_output.csv', index=False)
    return dfSchedule


# returns a dictionary storing all incoming edges in the final schedule graph
def get_incoming_edges(self):
    # initialize the dictionary
    incomingEdges = {}
    for n in self.vertices.keys():
        incomingEdges[n] = []

    # for each node, check the outgoing edges and add them to the respective entry in the dictionary
    for n in self.vertices.keys():
        n_d_e = self.dummy_edges[n]
        n_p_e = self.pre_edges[n]
        n_m_e = self.machine_edges[n]
        outgoingEdges = n_d_e | n_p_e | n_m_e
        for m in outgoingEdges:
            print(m)
            print(incomingEdges[m])
            incomingEdges[m].append(n)
    return incomingEdges


class Graph:
    def __init__(self, vertices, machines, dummy_edges, pre_edges, machine_edges):
        self.vertices = vertices
        self.machines = machines
        self.dummy_edges = dummy_edges
        self.pre_edges = pre_edges
        self.machine_edges = machine_edges

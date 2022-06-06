import importlib

import random
import numpy as np
import pandas as pd
import sys

from classes import milp_utils
from classes.milp import FlexibleJobShop


def create_graph(alg, v1, v2, v3):
    jobs = alg.jobs
    # create list of vertices containing of all jobs, and the 2 dummy nodes
    vertices = {-2: 0, -1: 0}
    # create a dict with key as vertex id and the value as weight
    for j in range(len(v3)):
        vertices.update({
            j: 0})
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
        if i + 1 > len(v3)-1:
            dummy_edges.update({i: {-1: 0}})
        elif v3[i + 1] == 0:
            dummy_edges.update({i: {-1: 0}})

        # statement for the precedence job edges
        if i + 1 < len(v3) and v3[i + 1] != 0:
            edges = {i + 1: 0}
            pre_edges.update({i: edges})
    dummy_edges.update({-2: dummies})
    # lastly fill the graph with machine relations
    return Graph(vertices, {}, dummy_edges, pre_edges, {})


def create_machine_edges(alg, v1, v3, g):
    machines = {}
    for m in alg.machines:
        machines.update({m: []})
    # create a dictionary where all operations that make use of a machine are listed
    for i in range(len(v1)):
        operations = machines.get(v1[i])
        operations.append(i)
        machines.update({v1[i]: operations})
    prev_job = -1
    machine_edges = get_machine_edges(alg, g, v3, machines)

    return Graph(g.vertices, machines, g.dummy_edges, g.pre_edges, machine_edges)


def k_insertion(alg, g, v3, v, pos, path):
    m_e = g.machine_edges.copy()
    m_e.update({v: {}})

    # Assign new machine, determine order and set weight
    machines = {}
    for m in alg.machines:
        machines.update({m: g.machines[m].copy()})
    # for i in range(len(v1)):
    #     operations = machines.get(v1[i])
    #     operations.append(i)
    #     machines.update({v1[i]: operations})
    for k in machines.keys():
        k_list = m_e.get(k)
        k_keys = k_list.keys()
        if v in k_keys:
            m_e.get(k).pop(v)
            break
    for k in machines:
        if v in machines[k]:
            machines[k].remove(v)
    g_min = Graph(g.vertices.copy(), machines, g.dummy_edges.copy(), g.pre_edges.copy(), m_e.copy())
    j = get_job_op(v3, v)
    o = v3[v]
    m_a = alg.machineAlternatives[j, o].copy()
    # Figure out how to determine m
    m = m_a[pos]
    qk = machines.get(m).copy()
    rk = set()
    lk = set()
    df = get_data_frame(g, alg, v3)
    cm = milp_utils.calculate_makespan(df)
    #s_v = get_start_time(g_min, v)
    s_v = 0
    t_v = cm
    #t_v = get_completion_time(g_min, cm, v)
    for x in qk:
        j_x = get_job_op(v3, x)
        # s_x = get_start_time(g, x)
        # s_x = get_critical_path(g, -2, x)
        s_x = 0
        # t_x = get_critical_path(g, x, -1)
        t_x = cm
        p_x = alg.processingTimes[j_x, v3[x], m]
        if s_x + p_x > s_v:
            rk.add(x)
        if p_x + t_x > t_v:
            lk.add(x)
    rk_lk = set(lk) & set(rk)
    upper_bound = 0
    if len(rk_lk) > 0:
        index = len(lk - rk) + random.randrange(len(rk_lk))
        qk.insert(index, v)
    elif len(rk_lk) == 0:
        q_k = set(qk.copy())
        q_k = q_k - (lk | rk)
        if len(q_k) > 0:
            index = len(lk - rk) + random.randrange(len(q_k))
            qk.insert(index, v)

        else:
            qk.insert(len(lk) - 1, v)
    machines.update({m: qk})
    m_e = get_machine_edges(alg, g, v3, machines)
    g_end = Graph(g_min.vertices.copy(), machines, g_min.dummy_edges.copy(), g_min.pre_edges.copy(), m_e.copy())
    return g_end


def get_machine_edges(alg, g, v3,  machines):
    machine_edges = {}
    for v in g.vertices.keys():
        machine_edges.update({v: {}})
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
    return machine_edges


def get_start_time(g, v):
    edges = get_incoming_edges(g)
    v_po_l = list(edges.get(v).keys())
    max_m = 0
    for i in v_po_l:
        if i == -2:
            return max_m
        else:
            s_j = get_start_time(g, i) + g.vertices.get(i) + edges.get(v).get(i)
            if s_j > max_m:
                max_m = s_j
    return max_m


def get_completion_time(g, cm, v):
    edges = get_outgoing_edges(g)
    v_so_l = list(edges.get(v).keys())
    min_m = cm
    for i in v_so_l:
        if i == -1:
            return min_m
        else:
            s_j = get_completion_time(g, cm, i) - g.vertices.get(i) - edges.get(v).get(i)
            if s_j < min_m:
                min_m = s_j
    return min_m


def get_job_op(v3, v):
    j = -1
    for x in range(len(v3)):
        if v3[x] == 0:
            j += 1
        if v == x:
            break
    return j


def get_critical_path(g, c, t):
    edges = get_outgoing_edges(g)
    visited = {}
    for i in g.vertices.keys():
        visited.update({i: False})
    path = []
    mkspns = []
    paths = []
    get_path(g, edges, c, t, visited, path, paths, 0, mkspns)
    max_m = max(mkspns)
    index = mkspns.index(max_m)
    c_p = paths[index]
    return max_m, c_p


def get_path(g, edges, c, t, visited, path, paths, m, max_m):
    visited.update({c: True})
    path.append(c)
    if c == t:
        max_m.append(m)
        paths.append(path.copy())
    else:
        m += g.vertices.get(c)
        for i in edges.get(c):
            test = edges.get(c)
            if not visited.get(i):
                co = test.get(i)
                m += co
                get_path(g, edges, i, t, visited, path, paths, m, max_m)

    path.pop()
    visited.update({c: False})


def get_data_frame(self, inst, v3):
    incomingEdges = get_incoming_edges_list(self)

    visited = {}  # keeps track if a node has been visited or not
    for i in self.vertices.keys():
        visited.update({i: False})
    visited.update({-2: True})
    timeArray = []
    lastEnzyme = []
    previousOpCompletion = []
    for i in range(len(inst.machines)):
        timeArray.append(0)  # last use time of each machine filled with 0
        lastEnzyme.append("")  # last enzyme used on each machine # empty string
    for i in range(len(inst.jobs)):
        previousOpCompletion.append(0)  # completion time of last operation on each job #Fileld with 0

    opsPerJobAcc = [0]  # list to easily find which job an operation node belongs to
    for j in range(len(inst.jobs)):
        opsInJob = len(inst.operations[inst.jobs[j]])
        opsPerJobAcc.append(opsPerJobAcc[j] + opsInJob)

    results = []
    l = -1
    queue = []  # a queue to keep track of the next node to visit
    for t in self.dummy_edges.get(-2).keys():
        queue.append(t)

    while len(queue) != 0:
        n = queue.pop(0)

        if n == l or visited[n]:
            continue

        skipping = False
        for i in (incomingEdges[n]):
            if not visited[i]:
                skipping = True
                break
        if skipping:
            continue

        # find out which job the node belongs to
        j = get_job_op(v3, n)
        # find the correct operation in that job
        o = v3[n]

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
        if lastEnzyme[m] != "":
            changeOver = inst.changeOvers[(m, lastEnzyme[m], p)]
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
        for t in self.dummy_edges.get(n).keys():
            queue.append(t)
        for t in self.pre_edges.get(n).keys():
            queue.append(t)
        for t in self.machine_edges.get(n).keys():
            queue.append(t)

    dfSchedule = pd.DataFrame(results)
    dfSchedule.sort_values(by=['Start', 'Machine', 'Job'], inplace=True)
    dfSchedule.to_csv('csv_output.csv', index=False)
    return dfSchedule


# returns a dictionary storing all incoming edges in the final schedule graph
def get_incoming_edges_list(self):
    # initialize the dictionary
    incomingEdges = {}
    for n in self.vertices.keys():
        incomingEdges[n] = []

    # for each node, check the outgoing edges and add them to the respective entry in the dictionary
    for n in self.vertices.keys():
        n_d_e = self.dummy_edges.get(n)
        n_p_e = self.pre_edges.get(n)
        n_m_e = self.machine_edges.get(n)
        outgoingEdges = n_d_e | n_p_e | n_m_e
        for m in outgoingEdges:
            incomingEdges[m].append(n)
    return incomingEdges


def get_incoming_edges(g):
    incoming_edges = {}
    for n in g.vertices.keys():
        incoming_edges.update({n: {}})
    for n in g.vertices.keys():
        n_d_e = g.dummy_edges[n]
        n_p_e = g.pre_edges[n]
        n_m_e = g.machine_edges[n]
        outgoing_edges = n_d_e | n_p_e | n_m_e
        for m in outgoing_edges.keys():
            incoming_edges.get(m).update({n: outgoing_edges.get(m)})
    return incoming_edges


def get_outgoing_edges(g):
    edges = {}
    for k in g.vertices.keys():
        e = g.dummy_edges.get(k).copy()
        e.update(g.pre_edges.get(k).copy())
        e.update(g.machine_edges.get(k).copy())
        edges.update({k: e})
    return edges


class Graph:
    def __init__(self, vertices, machines, dummy_edges, pre_edges, machine_edges):
        self.vertices = vertices
        self.machines = machines
        self.dummy_edges = dummy_edges
        self.pre_edges = pre_edges
        self.machine_edges = machine_edges

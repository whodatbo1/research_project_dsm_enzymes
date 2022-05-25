import importlib

import random
import numpy as np

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
    return Graph(vertices,{}, dummy_edges, pre_edges, {})


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
    for m in range(len(machines)):
        ops = machines.get(m)
        prev_job = -1
        for i in range(len(ops)):
            j = -1
            for x in range(len(v3)):
                if v3[x] == 0:
                    j += 1
                if ops[i] == x:
                    break
            weight = alg.processingTimes[j, v3[ops[i]], m]
            g.vertices.update({ops[i]: weight})
            if i + 1 < len(ops):
                co = 0
                if prev_job > 0:
                    co = alg.changeOvers[m, alg.orders[prev_job].get("product"), alg.orders[j].get("product")]
                machine_edges.update({ops[i]: {ops[i + 1]: co}})
            prev_job = j
    return Graph(g.vertices, machines, g.dummy_edges, g.pre_edges, machine_edges)


def k_insertion(alg, g, v3, node):
    # Remove weight from the node, and all machine edges to the node
    v = g.vertices.copy()
    v.update({node: 0})
    m_e = g.machine_edges.copy()
    for k in m_e.keys():
        m_e.get(k).pop(node, None)

    # Assign new machine, determine order and set weight
    machines = g.machines.copy()
    for k in machines.keys():
        if node in m_e.get(k).keys():
            m_e.get(k).remove(node)
            break
    j = -1
    for x in range(node):
        if v3[x] == 0:
            j += 1
    if node == 0:
        j = 0
    m = random.choice(alg.machineAlternatives[j, v3[node]])
    v.update({node: alg.processingTimes[j, v3[node], m]})
    ops = machines.get(m).copy()
    ops.insert(random.randrange(0, len(ops)), node)
    machines.update({m: ops})
    prev_job = -1
    for i in range(len(ops)):
        j = -1
        for x in range(len(v3)):
            if v3[x] == 0:
                j += 1
            if ops[i] == x:
                break
        if ops[i] == node and i+1 < len(ops):
            co = 0
            if prev_job > 0:
                co = alg.changeOvers[m, alg.orders[prev_job].get("product"), alg.orders[j].get("product")]
            m_e.update({ops[i]: {ops[i + 1]: co}})
        if ops[i-1] == node:
            co = 0
            if prev_job > 0:
                co = alg.changeOvers[m, alg.orders[prev_job].get("product"), alg.orders[j].get("product")]
            m_e.update({ops[i-1]: {i: co}})
        prev_job = j
    return Graph(v, machines,  g.dummy_edges, g.pre_edges, m_e)


def get_critical_path(alg, g):
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
    m = []
    get_path(g, edges, -2, -1, visited, path, 0, m)
    return max(m)


def get_path(g, edges, c, t, visited, path, m, max_m):
    visited.update({c: True})
    path.append(c)
    m += g.vertices.get(c)
    # also add changeover
    if c == t:
        max_m.append(m)
    else:

        for i in edges.get(c):
            test = edges.get(c)
            if not visited.get(i):
                co = test.get(i)
                m += co
                get_path(g, edges, i, t, visited, path, m, max_m)

    path.pop()
    visited.update({c: False})


class Graph:
    def __init__(self, vertices, machines, dummy_edges, pre_edges, machine_edges):
        self.vertices = vertices
        self.machines = machines
        self.dummy_edges = dummy_edges
        self.pre_edges = pre_edges
        self.machine_edges = machine_edges

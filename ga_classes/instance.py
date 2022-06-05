import numpy as np
import pandas as pd


def construct_v3(instance):
    v3 = []
    for job in instance.operations:
        for i in range(len(instance.operations[job])):
            v3.append(i)
    v3 = np.array(v3, dtype=np.uint64)
    return v3


def construct_job_vector(instance, v3):
    job_vector = np.zeros(len(v3), dtype=np.int64)
    index = 0
    for job in instance.jobs:
        op_count = len(instance.operations[job])
        job_vector[index:(index + op_count)] = np.full(op_count, job)
        index = index + op_count
    return job_vector


class Instance:

    def __init__(self, instance):
        self.v3 = construct_v3(instance)
        self.job_vector = construct_job_vector(instance, self.v3)
        self.nr_machines = instance.nr_machines
        self.nr_jobs = instance.nr_jobs
        self.orders = instance.orders
        self.machines = instance.machines
        self.jobs = instance.jobs
        self.operations = instance.operations
        self.machine_alternatives = instance.machineAlternatives
        self.processing_times = instance.processingTimes
        self.change_overs = instance.changeOvers
        self.instance = instance
        self.vector_length = sum([len(k) for k in self.operations.values()])
        self.init_indices = np.where(self.v3 == 0)[0]

    def generate_random_schedule_encoding(self):
        v_length = self.vector_length
        v = np.zeros((v_length, 2)).astype(np.uint64)

        index = 0
        for job in self.operations:
            for i in range(len(self.operations[job])):
                r = np.random.choice(self.machine_alternatives[job, i])
                v[index, 0] = r
                v[index, 1] = job
                index += 1

        counts = {j: 0 for j in self.jobs}
        max_counts = {j: len(self.operations[j]) for j in self.jobs}
        index = 0
        while index < v_length:
            rnint = np.random.randint(self.nr_jobs)
            if counts[rnint] < max_counts[rnint]:
                v[index, 1] = rnint
                counts[rnint] += 1
                index += 1

        v1 = v[:, 0]
        v2 = v[:, 1]

        return v1, v2

    def generate_random_schedule_encoding_zhang(self):
        v2 = self.job_vector.copy()
        np.random.shuffle(v2)
        op_counts = {j: 0 for j in self.jobs}

        v1 = np.zeros_like(v2)
        for job in v2:
            curr_op = op_counts[job]

            machine_1, machine_2 = np.random.choice(self.machine_alternatives[job, curr_op], 2)

            if self.processing_times[(job, curr_op, machine_1)] <= self.processing_times[(job, curr_op, machine_2)]:
                min_m = machine_1
            else:
                min_m = machine_2

            v1[self.init_indices[job] + op_counts[job]] = min_m

            op_counts[job] += 1

        return v1, v2


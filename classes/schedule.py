from gurobipy import *
import pandas as pd

class Schedule:
    def __init__(
            self, makespan, machines_operations, job_order):
        self.makespan = makespan
        self.machines_operations = machines_operations
        self.job_order = job_order



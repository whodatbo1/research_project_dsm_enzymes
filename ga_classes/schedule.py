import numpy as np
import pandas as pd
from main.ga_classes.instance import Instance
from main.ga_utils.encode_decode import encode_schedule, decode_schedule_active


class Schedule:

    def __init__(self, instance: Instance):
        self.instance: Instance = instance
        self.v1: np.ndarray = None
        self.v2: np.ndarray = None
        self.makespan: int = None
        self.total_machine_workload: int = None
        self.max_machine_workload: int = None
        self.latency: int = None
        self.schedule_df: pd.DataFrame = None

    def construct_from_df(self, df: pd.DataFrame):
        v1, v2 = encode_schedule(df)
        schedule, v1_active, v2_active = decode_schedule_active(self.instance, v1, v2, self.instance.v3)
        self.v1 = v1_active
        self.v2 = v2_active
        self.schedule_df = schedule
        self.calculate_all_fitness_values()
        return self

    def construct_from_vectors(self, v1, v2):
        schedule, v1_active, v2_active = decode_schedule_active(self.instance, v1, v2, self.instance.v3)
        self.schedule_df = schedule
        self.v1 = v1_active
        self.v2 = v2_active
        self.calculate_all_fitness_values()
        return self

    def calculate_all_fitness_values(self):
        self.makespan = calculate_makespan(self)
        self.total_machine_workload = calculate_total_machine_workload(self)
        self.max_machine_workload = calculate_max_machine_workload(self)
        self.latency = calculate_latency(self)

    # Returns all fitness values of schedule
    def get_fitness(self):
        return [self.makespan, self.total_machine_workload, self.max_machine_workload, self.latency]

    def __lt__(self, other):
        if not isinstance(other, Schedule):
            return NotImplemented

# Calculates the makespan of the schedule
def calculate_makespan(schedule: Schedule):
    return schedule.schedule_df["Completion"].max()


# Calculates the total latency of the schedule
# If the completion time of a job is <= the due time, there is no delay
# If it is larger, the delay = (completion time of the job) - (due time)
def calculate_latency(schedule: Schedule) -> int:
    maxes = schedule.schedule_df.groupby(by="Job", sort=False)["Completion"].max()
    total_delay = 0

    for job in schedule.instance.jobs:
        max_completion = maxes[job]
        due_date = schedule.instance.orders[job]['due']
        total_delay += max(0, max_completion - due_date)

    return total_delay


# Calculates total processing time across all machines
def calculate_total_machine_workload(schedule: Schedule) -> int:
    total_processing_time = schedule.schedule_df["Duration"].sum()
    return total_processing_time


# Calculates the maximum total processing time across all machines
def calculate_max_machine_workload(schedule: Schedule) -> int:
    max_machine_processing_time = schedule.schedule_df.groupby(by="Machine")["Duration"].sum().max()
    return max_machine_processing_time


# Compares 2 schedules to see if one dominates the other
# In all of the fitness values implemented now, lower is better
# If all of the fitness values of the first schedule are better - return 1
# If all of the fitness values of the second schedule are better - return -1
# Else - return 0
def compare_schedule(self, schedule_0: Schedule, schedule_1: Schedule) -> int:
    if schedule_0.makespan < schedule_1.makespan and \
            schedule_0.total_machine_workload < schedule_1.total_machine_workload and \
            schedule_0.max_machine_workload < schedule_1.max_machine_workload and \
            schedule_0.latency < schedule_1.latency:
        return 1
    elif schedule_0.makespan > schedule_1.makespan and \
            schedule_0.total_machine_workload > schedule_1.total_machine_workload and \
            schedule_0.max_machine_workload > schedule_1.max_machine_workload and \
            schedule_0.latency > schedule_1.latency:
        return -1
    return 0

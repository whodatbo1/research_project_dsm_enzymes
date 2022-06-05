import importlib.util
import pandas as pd
import os
from main.ga_classes.instance import Instance


def get_instance_info(instance_num: int) -> Instance:
    fileName = 'FJSP_' + str(instance_num)
    spec = importlib.util.spec_from_file_location('instance', os.getcwd() + "/instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return Instance(mod)


def calculate_problem_size(instance):
    init = 1
    for j in instance.jobs:
        for op in instance.operations[j]:
            init *= len(instance.machineAlternatives[j, op])
    return init


def check_valid(schedule: pd.DataFrame):
    s = schedule.copy(deep=True)
    s.sort_values(by=['Machine', 'Completion'], ignore_index=True, inplace=True)
    starts = pd.concat([pd.Series([0]), s['Completion']], ignore_index=True)

    s['Diff'] = s['Start'] - starts
    s['match'] = s.Machine.eq(s.Machine.shift(fill_value=0))
    s['check'] = (s.match * s.Diff) >= -0.1

    c = s.check.all()

    if not c:
        print(c)
        print(s)

    return c


#
# def validate_schedule(schedule: pd.DataFrame, instance):
#     schedule = schedule.copy(deep=True)
#     schedule.sort_values(by=['Job', 'Operation'], ignore_index=True, inplace=True)
#
#     groups = schedule.groupby(by='Job')
#
#     # Assumed to be working
#     for index, group in groups:
#         starts = group['Start'].to_numpy()
#         comps = group['Completion'].to_numpy()
#         potentially_sorted_times = np.dstack((starts, comps)).flatten()
#
#         if sorted(potentially_sorted_times) != potentially_sorted_times:
#             return False
#
#     schedule.sort_values(by=['Machine', 'Start'], ignore_index=True, inplace=True)
#
#     groups = schedule.groupby(by='Job')
#
#     for index, group in groups:
#         starts = group['Start'].to_numpy()
#         for start in range(len(starts) - 1):
#
#
#     return True


# Checks whether a schedule is feasiblie
# Code made by Robin
def feasibleSchedule(inst, df: pd.DataFrame):
    df.sort_values(by=["Start"], inplace=True)
    jobOps = []
    jobCompletion = []
    for i in range(inst.nr_jobs):
        jobOps.append(0)
        jobCompletion.append(0)
    lastEnzyme = []
    endingTime = []
    for i in range(inst.nr_machines):
        lastEnzyme.append("")
        endingTime.append(0)

    infeasible = False
    for row in df.values:
        m = row[0]
        j = row[1]
        p = row[2]
        o = row[3]
        s = row[4]
        d = row[5]
        c = row[6]

        # check if operations of one job are performed in order
        if o != jobOps[j] or jobCompletion[j] > s:
            infeasible = True

        # check if the operation is performed on a viable machine
        if not m in inst.machineAlternatives[(j, o)]:
            infeasible = True

        # check if completion time and duration are correct
        if s + d != c or d != inst.processingTimes[(j, o, m)]:
            infeasible = True

        # check if changeOver times are fulfilled
        changeOver = 0
        if lastEnzyme[m] != "":
            changeOver = inst.changeOvers[(m, lastEnzyme[m], p)]
        if endingTime[m] + changeOver > s:
            infeasible = True

        # update trackers
        jobOps[j] += 1
        jobCompletion[j] = c
        lastEnzyme[m] = p
        endingTime[m] = c

        if infeasible:
            return False
    return True


# calculate_latency()
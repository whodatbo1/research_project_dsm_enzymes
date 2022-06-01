import importlib.util
import pandas as pd

def get_instance_info(i):
    fileName = 'FJSP_' + str(i)
    spec = importlib.util.spec_from_file_location('instance', "instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

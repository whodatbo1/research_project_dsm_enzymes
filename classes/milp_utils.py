import pandas as pd
import importlib.util


def get_instance_info(i):
    fileName = 'FJSP_' + str(i)
    spec = importlib.util.spec_from_file_location('instance', "../instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def calculate_makespan(schedule):
    comp_times = schedule["Completion"]
    return comp_times.max()


# Do we have info for this btw?
def calculate_tardiness():
    pass


def calculate_idle_time(schedule: pd.DataFrame):
    # Sort by Machine and Start time, reset the indexing of DF

    fileName = 'FJSP_' + str(0)
    spec = importlib.util.spec_from_file_location('instance', "../instances/" + fileName + '.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    co_times = mod.changeOvers

    schedule.sort_values(by=['Machine', 'Start'], inplace=True, ignore_index=True)

    idle_times = {m:0 for m in range(schedule["Machine"].max() + 1)}

    for i in range(1, len(schedule)):
        curr = schedule.at[i, "Machine"]
        prev = schedule.at[i - 1, "Machine"]
        if curr == prev:
            # If it's the same machine, update the idle time
            idle_times[curr] += schedule.at[i, "Start"] - schedule.at[i - 1, "Completion"]\
                                - co_times[curr, schedule.at[i - 1, "Product"], schedule.at[i, "Product"]]

    # Return sum of idle times accross all machines
    return sum(idle_times.values())


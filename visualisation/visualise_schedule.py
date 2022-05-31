import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualise_schedule(schedule: pd.DataFrame):
    cols = plt.cm.rainbow(np.linspace(0, 1, max(schedule["Job"] + 1)))
    plt.figure()
    groups = schedule.groupby('Job')
    for name, group in groups:
        for i, entry in group.iterrows():
            points = [[entry["Start"], entry["Completion"]], [entry["Machine"], entry["Machine"]]]
            plt.plot(points[0], points[1], linewidth=10, c=cols[entry["Job"]], solid_capstyle="butt",
                     marker='d', markerfacecolor='black')
            plt.text((points[0][0] + points[0][1]) / 2 - 0.2, points[1][0] - 0.1, str(entry["Job"]))
    plt.xticks(np.arange(0, max(schedule["Completion"] + 1)))
    plt.grid()
    plt.title('Schedule')
    plt.xlabel('Time')
    plt.ylabel('Machine')
    plt.show()


sched = pd.read_csv('../solutions/milp/milp_solution_FJSP_2.csv')
visualise_schedule(sched)

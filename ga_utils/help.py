# from utils import *
import pandas as pd


def check_valid(schedule: pd.DataFrame):
    # s = pd.read_csv('./min_csv_output.csv')
    s = schedule.copy(deep=True)
    s.sort_values(by=['Machine', 'Completion'], ignore_index=True, inplace=True)
    starts = pd.concat([pd.Series([0]), s['Completion']], ignore_index=True)
    # starts.insert(0, 'Start', 0)
    # starts.append(s['Start'])
    s['Diff'] = s['Start'] - starts
    s['match'] = s.Machine.eq(s.Machine.shift(fill_value=0))
    s['check'] = (s.match * s.Diff) >= -0.1
    # for i, row in s.iterrows():
    # grouped_s = s.groupby("Machine")
    # print(s)
    c = s.check.all()
    if not c:
        print(c)
        print(s)
    return c


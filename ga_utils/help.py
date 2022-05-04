# from utils import *
import pandas as pd

s = pd.read_csv('./min_csv_output.csv')
s.sort_values(by=['Machine', 'Completion'], ignore_index=True, inplace=True)
print(s['Completion'])
starts = pd.concat([pd.Series([0]), s['Completion']], ignore_index=True)
print(starts)
# starts.insert(0, 'Start', 0)
# starts.append(s['Start'])
s['Diff'] = s['Start'] - starts
print(s)


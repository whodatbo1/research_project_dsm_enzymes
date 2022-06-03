import numpy as np
import pandas as pd
from main.ga_classes.instance import Instance
from main.ga_utils.encode_decode import encode_schedule, decode_schedule_active
from main.ga_utils.utils import calculate_makespan


class Schedule:

    def __init__(self, instance: Instance):
        self.instance: Instance = instance
        self.v1: np.ndarray = None
        self.v2: np.ndarray = None
        self.makespan: int = None
        self.schedule_df: pd.DataFrame = None

    def construct_from_df(self, df: pd.DataFrame):
        v1, v2 = encode_schedule(df)
        self.v1 = v1
        self.v2 = v2
        self.schedule_df = df
        self.makespan = calculate_makespan(df)
        return self

    def construct_from_vectors(self, v1, v2):
        schedule, v1_active, v2_active = decode_schedule_active(self.instance, v1, v2, self.instance.v3)
        self.schedule_df = schedule
        self.v1 = v1_active
        self.v2 = v2_active
        self.makespan = calculate_makespan(schedule)
        return self

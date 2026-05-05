import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

class Patient:
    def __init(self, p_id):
        self.id = p_id

        self.q_time_reg = pd.NA
        self.q_time_triage = pd.NA
        self.q_time_treat = pd.NA
        self.q_time_pharmacy = pd.NA


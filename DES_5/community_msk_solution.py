import simpy
from sim_tools.distributions import Poisson
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from tqdm import tqdm

os.chdir(Path(__file__).parent)

class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_assessment_apts = {}
        self.q_time_physio_apts = {}
        self.q_time_injection_apts = {}

class Param:
    def __init__(
        self,
        transition_prob_matrix_csv,
        num_assessment_slots_per_day = 10,
        num_physio_slots_per_day = 10,
        num_injection_slots_per_day = 10,
        mean_referrals_per_day = 10.0,
        results_collection_period = (365) * 5,
        warm_up_period = 365,
        num_replications = 100
    ):
        self.transition_prob_matrix_df = (
            pd.read_csv(transition_prob_matrix_csv)
        )
        self.num_assessment_slots_per_day = num_assessment_slots_per_day
        self.num_physio_slots_per_day = num_physio_slots_per_day
        self.num_injection_slots_per_day = num_injection_slots_per_day
        self.mean_referrals_per_day = mean_referrals_per_day
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.num_replications = num_replications


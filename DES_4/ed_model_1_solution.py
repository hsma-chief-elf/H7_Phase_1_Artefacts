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

class Param:
    def __init__(
        self,
        mean_patient_inter = 10,
        mean_reg_time = 3,
        sd_reg_time = 1,
        mean_triage_time = 10,
        sd_triage_time = 2,
        mean_treat_time = 120,
        sd_treat_time = 30,
        mean_pharm_time = 4,
        sd_pharm_time = 1,
        num_receptionists = 2,
        num_nurses = 2,
        num_doctors = 3,
        num_pharmacists = 1,
        results_collection_period = 2880,
        warm_up_period = 20,
        num_replications = 100,
        num_replications_warm_up_assessment = 50,
        warm_up_assessment_sim_length_scaler = 20,
        cumulative_mean_tracker_interval = 5
    ):
        self.mean_patient_inter = mean_patient_inter
        self.mean_reg_time = mean_reg_time
        self.sd_reg_time = sd_reg_time
        self.mean_triage_time = mean_triage_time
        self.sd_triage_time = sd_triage_time
        self.mean_treat_time = mean_treat_time
        self.sd_treat_time = sd_treat_time
        self.mean_pharm_time = mean_pharm_time
        self.sd_pharm_time = sd_pharm_time
        self.num_receptionists = num_receptionists
        self.num_nurses = num_nurses
        self.num_doctors = num_doctors
        self.num_pharmacists = num_pharmacists
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.sim_duration = warm_up_period + results_collection_period
        self.num_replications = num_replications
        self.num_replications_warm_up_assessment = (
            num_replications_warm_up_assessment
        )
        self.sim_duration_warm_up_assessment = (
            self.sim_duration * warm_up_assessment_sim_length_scaler
        )
        self.cumulative_mean_tracker_interval = (
            cumulative_mean_tracker_interval
        )


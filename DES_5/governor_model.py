import simpy
from sim_tools.distributions import Lognormal
from sim_tools.distributions import Poisson
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.num_follow_ups = pd.NA

class Param:
    def __init__(
        self,
        num_slots_per_day = 10,
        mean_referrals_per_day = 12.0,
        results_collection_period = 365,
        warm_up_period = 365,
        num_replications = 100,
        num_replications_warm_up_assessment = 50,
        warm_up_assessment_sim_length_scaler = 20,
        cumulative_mean_tracker_interval = 1
    ):
        self.num_slots_per_day = num_slots_per_day
        self.mean_referrals_per_day = mean_referrals_per_day
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.sim_duration = warm_up_period + results_collection_period
        self.num_replications = num_replications
        self.num_replications_warm_up_assessment = (
            num_replications_warm_up_assessment
        )
        self.warm_up_assessment_sim_length_scaler = (
            warm_up_assessment_sim_length_scaler
        )
        self.cumulative_mean_tracker_interval = (
            cumulative_mean_tracker_interval
        )

class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0

        self.daily_slots = simpy.Container(
            env,
            self.param.num_slots_per_day,
            init=self.param.num_slots_per_day
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(1)

        self.referrals_per_day_dist = Poisson(
            mean=self.param.mean_referrals_per_day,
            random_seed=seeds[0]
        )

        self.list_of_patients = []

    def generator_new_referrals(self):
        while True:
            todays_referrals = self.referrals_per_day_dist.sample()
            print (f"Day {self.env.now}: {todays_referrals} referrals")

            for referral in range(todays_referrals):
                self.patient_counter += 1
                p = Patient(self.patient_counter)
                self.list_of_patients.append(p)
                self.env.process(self.attend_first_apt(p))

            yield self.env.timeout(1)

    def attend_first_apt(self, patient):
        start_q_first_apt = self.env.now

        yield self.daily_slots.get(1)

        end_q_first_apt = self.env.now
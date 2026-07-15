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
        self.current_apt_type = "assessment"
        self.current_assessment_apt_id = 0
        self.current_physio_apt_id = 0
        self.current_injection_apt_id = 0
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
        fixed_delay_after_assessment = 7,
        fixed_delay_after_physio = 7,
        fixed_delay_after_injection = 42,
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
        self.fixed_delay_after_assessment = fixed_delay_after_assessment
        self.fixed_delay_after_physio = fixed_delay_after_physio
        self.fixed_delay_after_injection = fixed_delay_after_injection
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.num_replications = num_replications

class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0

        self.daily_assessment_slots = simpy.Container(
            self.env,
            self.param.num_assessment_slots_per_day,
            init=self.param.num_assessment_slots_per_day
        )

        self.daily_physio_slots = simpy.Container(
            self.env,
            self.param.num_physio_slots_per_day,
            init=self.param.num_physio_slots_per_day
        )

        self.daily_injection_slots = simpy.Container(
            self.env,
            self.param.num_injection_slots_per_day,
            init=self.param.num_injection_slots_per_day
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(4)

        self.referrals_per_day_dist = Poisson(
            rate=self.param.mean_referrals_per_day,
            random_seed=seeds[0]
        )

        self.from_assessment_transition_rng = (
            np.random.default_rng(seeds[1])
        )

        self.from_physio_transition_rng = (
            np.random.default_rng(seeds[2])
        )

        self.from_injection_transition_rng = (
            np.random.default_rng(seeds[3])
        )

        self.list_of_patients = []
        self.mean_q_time_assessment_apts = {}
        self.mean_q_time_physio_apts = {}
        self.mean_q_time_injection_apts = {}
        self.sd_q_time_assessment_apts = {}
        self.sd_q_time_physio_apts = {}
        self.sd_q_time_injection_apts = {}
        self.perc_90_q_time_assessment_apts = {}
        self.perc_90_q_time_physio_apts = {}
        self.perc_90_q_time_injection_apts = {}

    def generator_new_referrals(self):
        while True:
            todays_referrals = self.referrals_per_day_dist.sample()

            for referral in range(todays_referrals):
                self.patient_counter += 1
                p = Patient(self.patient_counter)
                self.list_of_patients.append(p)
                self.env.process(self.appointment_governor(p))

            yield self.env.timeout(1)

    def attend_assessment_apt(self, patient, is_first):
        start_q_assessment_apt = self.env.now

        if is_first:
            slots_to_consume = 1.0
        else:
            slots_to_consume = 0.5

        yield self.daily_assessment_slots.get(slots_to_consume)

        end_q_assessment_apt = self.env.now

        if self.env.now > self.param.warm_up_period:
            patient.q_time_assessment_apts[
                patient.current_assessment_apt_id
            ] = (
                end_q_assessment_apt - start_q_assessment_apt
            )

        yield self.env.timeout(1)

        yield self.daily_assessment_slots.put(slots_to_consume)

    
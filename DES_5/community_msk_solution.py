from numpy.char import startswith
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
        self.transition_prob_matrix_df = (
            self.transition_prob_matrix_df.set_index("current_state")
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
        seeds = ss.spawn(2)

        self.referrals_per_day_dist = Poisson(
            rate=self.param.mean_referrals_per_day,
            random_seed=seeds[0]
        )

        self.transition_rng = (
            np.random.default_rng(seeds[1])
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

    def attend_physio_apt(self, patient, is_first):
        start_q_physio_apt = self.env.now

        if is_first:
            slots_to_consume = 2.0
        else:
            slots_to_consume = 1.0

        yield self.daily_physio_slots.get(slots_to_consume)

        end_q_physio_apt = self.env.now

        if self.env.now > self.param.warm_up_period:
            patient.q_time_physio_apts[
                patient.current_physio_apt_id
            ] = (
                end_q_physio_apt - start_q_physio_apt
            )

        yield self.env.timeout(1)

        yield self.daily_physio_slots.put(slots_to_consume)

    def attend_injection_apt(self, patient):
        start_q_injection_apt = self.env.now

        slots_to_consume = 1.0

        yield self.daily_injection_slots.get(slots_to_consume)

        end_q_injection_apt = self.env.now

        if self.env.now > self.param.warm_up_period:
            patient.q_time_injection_apts[
                patient.current_injection_apt_id
            ] = (
                end_q_injection_apt - start_q_injection_apt
            )

        yield self.env.timeout(1)

        yield self.daily_injection_slots.put(slots_to_consume)

    def post_appointment_delay(self, patient, time_to_delay):
        yield self.env.timeout(time_to_delay)

    def appointment_governor(self, patient):
        # FIRST APPOINTMENT
        patient.current_assessment_apt_id += 1
        yield self.env.process(
            self.attend_assessment_apt(patient, True)
        )

        while True:
            # GRAB TRANSITION PROBABILITIES FROM CURRENT STATE
            transition_row = (
                self.param.transition_prob_matrix_df.loc[
                    patient.current_apt_type
                ]
            )

            # DECIDE NEXT ACTIVITY
            sampled_prob_comp = (
                self.transition_rng.random()
            )

            if sampled_prob_comp < transition_row["assessment"]:
                # DELAY
                if patient.current_apt_type == "assessment":
                    delay = self.param.fixed_delay_after_assessment
                elif patient.current_apt_type == "physio":
                    delay = self.param.fixed_delay_after_physio
                else:
                    delay = self.param.fixed_delay_after_injection

                yield self.env.process(self.post_appointment_delay(
                    patient, delay
                ))

                # APPOINTMENT
                patient.current_apt_type = "assessment"
                patient.current_assessment_apt_id += 1
                yield self.env.process(
                    self.attend_assessment_apt(
                        patient,
                        False
                    )
                )
            elif sampled_prob_comp < (
                transition_row["assessment"] + 
                transition_row["physio"]
            ):
                # DELAY
                if patient.current_apt_type == "assessment":
                    delay = self.param.fixed_delay_after_assessment
                elif patient.current_apt_type == "physio":
                    delay = self.param.fixed_delay_after_physio
                else:
                    delay = self.param.fixed_delay_after_injection

                yield self.env.process(self.post_appointment_delay(
                    patient, delay
                ))

                # APPOINTMENT
                patient.current_apt_type = "physio"
                patient.current_physio_apt_id += 1
                if patient.current_physio_apt_id > 1:
                    first = False
                else:
                    first = True
                yield self.env.process(
                    self.attend_physio_apt(
                        patient,
                        first
                    )
                )
            elif sampled_prob_comp < (
                transition_row["assessment"] +
                transition_row["physio"] +
                transition_row["injection"]
            ):
                # DELAY
                if patient.current_apt_type == "assessment":
                    delay = self.param.fixed_delay_after_assessment
                elif patient.current_apt_type == "physio":
                    delay = self.param.fixed_delay_after_physio
                else:
                    delay = self.param.fixed_delay_after_injection

                yield self.env.process(self.post_appointment_delay(
                    patient, delay
                ))

                # APPOINTMENT
                patient.current_apt_type = "injection"
                patient.current_injection_apt_id += 1
                yield self.env.process(
                    self.attend_injection_apt(
                        patient
                    )
                )
            else:
                # LEAVE SYSTEM
                return

    def run_model(self):
        self.env.process(self.generator_new_referrals())
        self.env.run(until=self.param.sim_duration)

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dataframe = pd.DataFrame(
            entity.__dict__ for entity in entity_list
        )

        entity_dataframe = (
            entity_dataframe.join(
                entity_dataframe["q_time_assessment_apts"]
                .apply(pd.Series)
                .add_prefix("assessment_")
            )
        )
        entity_dataframe = (
            entity_dataframe.join(
                entity_dataframe["q_time_physio_apts"]
                .apply(pd.Series)
                .add_prefix("physio_")
            )
        )
        entity_dataframe = (
            entity_dataframe.join(
                entity_dataframe["q_time_injection_apts"]
                .apply(pd.Series)
                .add_prefix("injection_")
            )
        )

        entity_dataframe.drop(
            columns=[
                "q_time_assessment_apts",
                "q_time_physio_apts",
                "q_time_injection_apts"
            ],
            inplace=True
        )

        return entity_dataframe

    def calculate_run_results(self, entity_dataframe):
        assessment_apt_cols = [
            col for col in entity_dataframe.columns
            if col.startswith("assessment_")
        ]
        physio_apt_cols = [
            col for col in entity_dataframe.columns
            if col.startswith("physio_")
        ]
        injection_apt_cols = [
            col for col in entity_dataframe.columns
            if col.startswith("injection_")
        ]

        for col in assessment_apt_cols:
            self.mean_q_time_assessment_apts[col] = (
                entity_dataframe[col].mean()
            )
            self.sd_q_time_assessment_apts[col] = (
                entity_dataframe[col].std()
            )
            self.perc_90_q_time_assessment_apts[col] = (
                entity_dataframe[col].quantile(0.9)
            )

        for col in physio_apt_cols:
            self.mean_q_time_physio_apts[col] = (
                entity_dataframe[col].mean()
            )
            self.sd_q_time_physio_apts[col] = (
                entity_dataframe[col].std()
            )
            self.perc_90_q_time_physio_apts[col] = (
                entity_dataframe[col].quantile(0.9)
            )

        for col in injection_apt_cols:
            self.mean_q_time_injection_apts[col] = (
                entity_dataframe[col].mean()
            )
            self.sd_q_time_physio_apts[col] = (
                entity_dataframe[col].std()
            )
            self.perc_90_q_time_physio_apts[col] = (
                entity_dataframe[col].quantile(0.9)
            )

class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_assessment_apts = {}
        self.trial_mean_q_time_physio_apts = {}
        self.trial_mean_q_time_injection_apts = {}
        self.trial_sd_q_time_assessment_apts = {}
        self.trial_sd_q_time_physio_apts = {}
        self.trial_sd_q_time_injection_apts = {}
        self.trial_perc_90_assessment_apts = {}
        self.trial_perc_90_physio_apts = {}
        self.trial_perc_90_injection_apts = {}
        self.ci_lower_q_time_assessment_apts = {}
        self.ci_lower_q_time_physio_apts = {}
        self.ci_lower_q_time_injection_apts = {}
        self.ci_upper_q_time_assessment_apts = {}
        self.ci_upper_q_time_physio_apts = {}
        self.ci_upper_q_time_injection_apts = {}
        self.se_q_time_assessment_apts = {}
        self.se_q_time_physio_apts = {}
        self.se_q_time_injection_apts = {}

    
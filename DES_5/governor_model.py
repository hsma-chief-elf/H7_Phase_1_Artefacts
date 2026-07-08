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
        self.q_time_first_apt = pd.NA
        self.q_time_fu_apts = {}
        self.current_apt_id = 0

class Param:
    def __init__(
        self,
        num_slots_per_day = 40,
        mean_referrals_per_day = 12.0,
        prob_next_apt_dict = {
            0:0.7,
            1:0.9,
            2:0.85,
            3:0.75,
            4:0.6,
            5:0.5,
            6:0.4,
            7:0.35,
            8:0.25,
            9:0.2,
            10:0.15,
            11:0.1,
            12:0.05
        },
        gap_between_fu_apts = 90,
        results_collection_period = (365 * 5),
        warm_up_period = 365,
        num_replications = 100,
        num_replications_warm_up_assessment = 50,
        warm_up_assessment_sim_length_scaler = 20,
        cumulative_mean_tracker_interval = 1
    ):
        self.num_slots_per_day = num_slots_per_day
        self.mean_referrals_per_day = mean_referrals_per_day
        self.prob_next_apt_dict = prob_next_apt_dict
        self.max_key_prob_next_apt_dict = max(prob_next_apt_dict)
        self.gap_between_fu_apts = gap_between_fu_apts
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
            self.env,
            self.param.num_slots_per_day,
            init=self.param.num_slots_per_day
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(2)

        self.referrals_per_day_dist = Poisson(
            rate=self.param.mean_referrals_per_day,
            random_seed=seeds[0]
        )

        self.follow_up_decider_rng = (
            np.random.default_rng(seeds[1])
        )

        self.list_of_patients = []
        self.mean_q_time_first_apt = pd.NA
        self.sd_q_time_first_apt = pd.NA
        self.perc_90_q_time_first_apt = pd.NA
        self.mean_q_time_fu_apts = {}
        self.sd_q_time_fu_apts = {}
        self.perc_90_q_time_fu_apts = {}

    def generator_new_referrals(self):
        while True:
            todays_referrals = self.referrals_per_day_dist.sample()
            #print (f"Day {self.env.now}: {todays_referrals} referrals")

            for referral in range(todays_referrals):
                self.patient_counter += 1
                p = Patient(self.patient_counter)
                self.list_of_patients.append(p)
                self.env.process(self.appointment_governor(p))

            yield self.env.timeout(1)

    def appointment_governor(self, patient):
        yield self.env.process(self.attend_first_apt(patient))

        while True:
            apt_id_clamp = min(
                patient.current_apt_id,
                self.param.max_key_prob_next_apt_dict
            )

            if (
                self.follow_up_decider_rng.random() <
                self.param.prob_next_apt_dict[apt_id_clamp]
            ):
                patient.current_apt_id += 1
                yield self.env.process(self.delay_until_apt_due(patient))
                yield self.env.process(self.attend_fu_apt(patient))

    def attend_first_apt(self, patient):
        start_q_first_apt = self.env.now
        
        yield self.daily_slots.get(1)

        print (f"Patient {patient.id} attending FIRST APPOINTMENT")
        
        end_q_first_apt = self.env.now

        if self.env.now > self.param.warm_up_period:
            patient.q_time_first_apt = end_q_first_apt - start_q_first_apt
        
        yield self.env.timeout(1)

        yield self.daily_slots.put(1)

    def delay_until_apt_due(self, patient):
        yield self.env.timeout(self.param.gap_between_fu_apts)

    def attend_fu_apt(self, patient):
        start_q_fu_apt = self.env.now

        yield self.daily_slots.get(1)

        print (
            f"Patient {patient.id} attending FU appointment",
            patient.current_apt_id
        )
        
        end_q_fu_apt = self.env.now

        if self.env.now > self.param.warm_up_period:
            patient.q_time_fu_apts[patient.current_apt_id] = (
                end_q_fu_apt - start_q_fu_apt
            )
            
        yield self.env.timeout(1)

        yield self.daily_slots.put(1)

    def run_model(self):
        self.env.process(self.generator_new_referrals())
        self.env.run(until=self.param.sim_duration)

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dataframe = pd.DataFrame(
            entity.__dict__ for entity in entity_list
        )

        entity_dataframe = (
            entity_dataframe.join(
                entity_dataframe["q_time_fu_apts"].apply(pd.Series)
            )
        )
        entity_dataframe.drop(columns=["q_time_fu_apts"], inplace=True)

        return entity_dataframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_first_apt = (
            entity_dataframe["q_time_first_apt"].mean()
        )
        self.sd_q_time_first_apt = (
            entity_dataframe["q_time_first_apt"].std()
        )
        self.perc_90_q_time_first_apt = (
            entity_dataframe["q_time_first_apt"].quantile(0.9)
        )

        for fu_num in range(1, self.param.max_key_prob_next_apt_dict + 1):
            if fu_num in entity_dataframe.columns:
                self.mean_q_time_fu_apts[fu_num] = (
                    entity_dataframe[fu_num].mean()
                )
                self.sd_q_time_fu_apts[fu_num] = (
                    entity_dataframe[fu_num].std()
                )
                self.perc_90_q_time_fu_apts[fu_num] = (
                    entity_dataframe[fu_num].quantile(0.9)
                )
            else:
                self.mean_q_time_fu_apts[fu_num] = pd.NA
                self.sd_q_time_fu_apts[fu_num] = pd.NA
                self.perc_90_q_time_fu_apts[fu_num] = pd.NA

class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_first_apt = pd.NA
        self.trial_sd_q_time_first_apt = pd.NA
        self.trial_perc_90_q_time_first_apt = pd.NA
        self.ci_lower_q_time_first_apt = pd.NA
        self.ci_upper_q_time_first_apt = pd.NA
        self.se_q_time_first_apt = pd.NA
        self.trial_mean_q_time_fu_apts = {}
        self.trial_sd_q_time_fu_apts = {}
        self.trial_perc_90_q_time_fu_apts = {}
        self.ci_lower_q_time_fu_apts = {}
        self.ci_upper_q_time_fu_apts = {}
        self.se_q_time_fu_apts = {}

    def run_trial(self):
        for replication_id in range(self.param.num_replications):
            model_replication = Model(self.param, replication_id)
            model_replication.run_model()
            patient_df = model_replication.convert_entity_list_to_dataframe(
                model_replication.list_of_patients
            )
            model_replication.calculate_run_results(patient_df)
            self.list_of_simulation_replications.append(model_replication)

    def calculate_trial_results(self):
        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in 
            self.list_of_simulation_replications
        )

        self.trial_mean_q_time_first_apt = (
            self.replication_df["mean_q_time_first_apt"].mean()
        )

        self.trial_sd_q_time_first_apt = (
            self.replication_df["mean_q_time_first_apt"].std()
        )

        self.trial_perc_90_q_time_first_apt = (
            self.replication_df["mean_q_time_first_apt"].quantile(0.9)
        )

        self.se_q_time_first_apt = (
            self.trial_sd_q_time_first_apt /
            math.sqrt(self.param.num_replications)
        )

        t = stats.t.ppf(0.975, df=self.param.num_replications - 1)

        self.ci_lower_q_time_first_apt = (
            self.trial_mean_q_time_first_apt - (t * self.se_q_time_first_apt)
        )

        self.ci_upper_q_time_first_apt = (
            self.trial_mean_q_time_first_apt + (t * self.se_q_time_first_apt)
        )

        fu_means = pd.DataFrame(
            [replication.mean_q_time_fu_apts
            for replication in self.list_of_simulation_replications]
        )

        fu_means = fu_means.dropna(axis=1, how="all")

        self.trial_mean_q_time_fu_apts = fu_means.mean().to_dict()
        self.trial_sd_q_time_fu_apts = fu_means.std().to_dict()
        self.trial_perc_90_q_time_fu_apts = fu_means.quantile(0.9).to_dict()

        fu_means_means = fu_means.mean()
        fu_sd_values = fu_means.std()
        n_fu_means = fu_means.count()
        fu_se_values = fu_sd_values / np.sqrt(n_fu_means)
        self.se_q_time_fu_apts = fu_se_values.to_dict()

        t = pd.Series(
            stats.t.ppf(0.975, df=n_fu_means - 1),
            index=n_fu_means.index
        )
        self.trial_ci_lower_q_time_fu_apts = (
            (fu_means_means - t * fu_se_values).to_dict()
        )
        self.trial_ci_upper_q_time_fu_apts = (
            (fu_means_means + t * fu_se_values).to_dict()
        )

base_case_params = Param(num_replications=10)
base_case_trial = Trial(base_case_params)
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()

print ("BASE CASE TRIAL RESULTS")
print ("-----------------------")
print ("Queuing Time for First Appointment")
print (f"Mean: {base_case_trial.trial_mean_q_time_first_apt:.2f} days")
print (f"SD: {base_case_trial.trial_sd_q_time_first_apt:.2f} days")
print (f"90th Perc: {base_case_trial.trial_perc_90_q_time_first_apt:.2f} days")
print (f"SE: {base_case_trial.se_q_time_first_apt:.2f} days")
print (
    f"95% CI : ({base_case_trial.ci_lower_q_time_first_apt:.2f}, ",
    f"{base_case_trial.ci_upper_q_time_first_apt:.2f}) days"
)

print ()

print ("Queuing Time for Follow-Up Appointments (beyond due date)")

for fu_num in sorted(base_case_trial.trial_mean_q_time_fu_apts):
    print (
        f"FU {fu_num} :",
        f"Mean : {base_case_trial.trial_mean_q_time_fu_apts[fu_num]:.2f}",
        f"SD : {base_case_trial.trial_sd_q_time_fu_apts[fu_num]:.2f}",
        f"90th P : {base_case_trial.trial_perc_90_q_time_fu_apts[fu_num]:.2f}",
        f"SE : {base_case_trial.se_q_time_fu_apts[fu_num]:.2f}",
        f"95% CI :",
        f"({base_case_trial.trial_ci_lower_q_time_fu_apts[fu_num]:.2f},",
        f"{base_case_trial.trial_ci_upper_q_time_fu_apts[fu_num]:.2f}) days"
    )


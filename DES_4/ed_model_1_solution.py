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
        branch_prob_triage_to_pharm = 0.4,
        branch_prob_treat_to_pharm = 0.25,
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
        self.branch_prob_triage_to_pharm = branch_prob_triage_to_pharm
        self.branch_prob_treat_to_pharm = branch_prob_treat_to_pharm
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

class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.receptionist = simpy.Resource(
            self.env,
            capacity=self.param.num_receptionists
        )
        self.nurse = simpy.Resource(
            self.env,
            capacity=self.param.num_nurses
        )
        self.doctor = simpy.Resource(
            self.env,
            capacity=self.param.num_doctors
        )
        self.pharmacist = simpy.Resource(
            self.env,
            capacity=self.param.num_pharmacists
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(7)
        self.patient_inter_dist = Exponential(
            mean=self.param.mean_patient_inter,
            random_seed=seeds[0]
        )
        self.reg_act_time_dist = Lognormal(
            mean=self.param.mean_reg_time,
            stdev=self.param.sd_reg_time,
            random_seed=seeds[1]
        )
        self.triage_act_time_dist = Lognormal(
            mean=self.param.mean_triage_time,
            stdev=self.param.sd_triage_time,
            random_seed=seeds[2]
        )
        self.treat_act_time_dist = Lognormal(
            mean=self.param.mean_treat_time,
            stdev=self.param.sd_treat_time,
            random_seed=seeds[3]
        )
        self.pharm_act_time_dist = Lognormal(
            mean=self.param.mean_pharm_time,
            stdev=self.param.sd_pharm_time,
            random_seed=seeds[4]
        )
        self.triage_pharm_branch_prob_rng = (
            np.random.default_rng(seeds[5])
        )
        self.treat_pharm_branch_prob_rng = (
            np.random.default_rng(seeds[6])
        )

        self.list_of_patients = []
        self.mean_q_time_reg = pd.NA
        self.sd_q_time_reg = pd.NA
        self.perc_90_q_time_reg = pd.NA
        self.mean_q_time_triage = pd.NA
        self.sd_q_time_triage = pd.NA
        self.perc_90_q_time_triage = pd.NA
        self.mean_q_time_treat = pd.NA
        self.sd_q_time_treat = pd.NA
        self.perc_90_q_time_treat = pd.NA
        self.mean_q_time_pharm = pd.NA
        self.sd_q_time_pharm = pd.NA
        self.perc_90_q_time_pharm = pd.NA

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.list_of_patients.append(p)
            self.env.process(self.attend_ed(p))
            sampled_inter = self.patient_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    def cumulative_mean_tracker(self):
        yield self.env.timeout(self.param.cumulative_mean_tracker_interval)

        self.cumulative_mean_df = pd.DataFrame(columns=["Simulation Time"])

        while True:
            if self.list_of_patients:
                df_patients = pd.DataFrame(
                    [vars(p) for p in self.list_of_patients]
                )

                row_of_means = df_patients.mean()
                row_of_means["Simulation Time"] = self.env.now
            else:
                row_of_means = pd.Series(
                    {"Simulation Time" : self.env.now}
                )

            self.cumulative_mean_df = pd.concat(
                [self.cumulative_mean_df, row_of_means.to_frame().T],
                ignore_index=True
            )

            yield self.env.timeout(self.param.cumulative_mean_tracker_interval)

    def attend_ed(self, patient):
        start_q_reg = self.env.now

        with self.receptionist.request() as req:
            yield req
            end_q_reg = self.env.now
            if self.env.now > self.param.warm_up_period:
                patient.q_time_reg = end_q_reg - start_q_reg
            sampled_reg_act_time = self.reg_act_time_dist.sample()
            yield self.env.timeout(sampled_reg_act_time)

        start_q_triage = self.env.now
        
        with self.nurse.request() as req:
            yield req
            end_q_triage = self.env.now
            if self.env.now > self.param.warm_up_period:
                patient.q_time_triage = end_q_triage - start_q_triage
            sampled_triage_act_time = self.triage_act_time_dist.sample()
            yield self.env.timeout(sampled_triage_act_time)

        if (
            self.triage_pharm_branch_prob_rng.random() <
            self.param.branch_prob_triage_to_pharm
        ):
            start_q_pharm = self.env.now

            with self.pharmacist.request() as req:
                yield req
                end_q_pharm = self.env.now
                if self.env.now > self.param.warm_up_period:
                    patient.q_time_pharmacy = end_q_pharm - start_q_pharm
                sampled_pharm_act_time = self.pharm_act_time_dist.sample()
                yield self.env.timeout(sampled_pharm_act_time)

            # SINK AFTER PHARMACY
        else:
            start_q_treat = self.env.now

            with self.doctor.request() as req:
                yield req
                end_q_treat = self.env.now
                if self.env.now > self.param.warm_up_period:
                    patient.q_time_treat = end_q_treat - start_q_treat
                sampled_treat_act_time = self.treat_act_time_dist.sample()
                yield self.env.timeout(sampled_treat_act_time)

            if (
                self.treat_pharm_branch_prob_rng.random() <
                self.param.branch_prob_treat_to_pharm
            ):
                start_q_pharm = self.env.now

                with self.pharmacist.request() as req:
                    yield req
                    end_q_pharm = self.env.now
                    if self.env.now > self.param.warm_up_period:
                        patient.q_time_pharmacy = end_q_pharm - start_q_pharm
                    sampled_pharm_act_time = self.pharm_act_time_dist.sample()
                    yield self.env.timeout(sampled_pharm_act_time)

                # SINK AFTER PHARMACY
            
    
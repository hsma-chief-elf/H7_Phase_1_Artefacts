import simpy
from sim_tools.distributions import Lognormal
from sim_tools.time_dependent import NSPPThinning
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

class Patient:
    def __init__(self, p_id, priority): # NEW
        self.id = p_id

        self.q_time_reg = pd.NA
        self.q_time_triage = pd.NA
        self.q_time_treat = pd.NA
        self.q_time_pharmacy = pd.NA
        
        self.arrival_time = pd.NA

        self.priority = priority # NEW

class Param:
    def __init__(
        self,
        patient_iat_csv,
        mean_reg_time = 4,
        sd_reg_time = 1.2,
        mean_pharm_time = 6,
        sd_pharm_time = 2,
        num_receptionists = 2, # NEW
        num_nurses = 6, # NEW
        num_doctors = 10, # NEW
        num_pharmacists = 2, # NEW
        results_collection_period = 2880,
        warm_up_period = 10000,
        num_replications = 40,
        num_replications_warm_up_assessment = 20,
        warm_up_assessment_sim_length_scaler = 10,
        cumulative_mean_tracker_interval = 100,
        nurse_unav_time = 480, # NEW
        nurse_unav_freq = 960, # NEW
        num_nurses_unav = 1, # NEW
        doctor_unav_time = 120, # NEW
        doctor_unav_freq = 240, # NEW
        num_doctors_unav = 3 # NEW
    ):
        self.mean_reg_time = mean_reg_time
        self.sd_reg_time = sd_reg_time
        # NEW
        self.mean_triage_time_dict = {
            1:20,
            2:16,
            3:10,
            4:8,
            5:7
        }
        # NEW
        self.sd_triage_time_dict = {
            1:7,
            2:5,
            3:2,
            4:3,
            5:1
        }
        # NEW
        self.mean_treat_time_dict = {
            1:240,
            2:100,
            3:45,
            4:20,
            5:15
        }
        # NEW
        self.sd_treat_time_dict = {
            1:90,
            2:60,
            3:20,
            4:10,
            5:5
        }
        self.mean_pharm_time = mean_pharm_time
        self.sd_pharm_time = sd_pharm_time
        self.num_receptionists = num_receptionists
        self.num_nurses = num_nurses
        self.num_doctors = num_doctors
        self.num_pharmacists = num_pharmacists
        # NEW
        self.branch_prob_triage_to_pharm_dict = {
            1:0.01,
            2:0.03,
            3:0.2,
            4:0.6,
            5:0.85
        }
        # NEW
        self.branch_prob_treat_to_pharm_dict = {
            1:0.9,
            2:0.7,
            3:0.6,
            4:0.3,
            5:0.25
        }
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

        self.pt_arrivals_time_dependent_df = (
            pd.read_csv(patient_iat_csv)
        )

        # NEW
        self.nurse_unav_time = nurse_unav_time
        self.nurse_unav_freq = nurse_unav_freq
        self.num_nurses_unav = num_nurses_unav
        self.doctor_unav_time = doctor_unav_time
        self.doctor_unav_freq = doctor_unav_freq
        self.num_doctors_unav = num_doctors_unav

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
        
        # NEW
        self.nurse = simpy.PriorityResource(
            self.env,
            capacity=self.param.num_nurses
        )
        # NEW
        self.doctor = simpy.PriorityResource(
            self.env,
            capacity=self.param.num_doctors
        )
        self.pharmacist = simpy.Resource(
            self.env,
            capacity=self.param.num_pharmacists
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(17) # NEW - added extra seed spawns

        self.patient_inter_dist = NSPPThinning(
            data=param.pt_arrivals_time_dependent_df,
            random_seed1=seeds[0],
            random_seed2=seeds[7]
        )

        self.reg_act_time_dist = Lognormal(
            mean=self.param.mean_reg_time,
            stdev=self.param.sd_reg_time,
            random_seed=seeds[1]
        )
        # NEW
        self.triage_act_time_dist_p1 = Lognormal(
            mean=self.param.mean_triage_time_dict[1],
            stdev=self.param.sd_triage_time_dict[1],
            random_seed=seeds[2]
        )
        # NEW
        self.triage_act_time_dist_p2 = Lognormal(
            mean=self.param.mean_triage_time_dict[2],
            stdev=self.param.sd_triage_time_dict[2],
            random_seed=seeds[9]
        )
        # NEW
        self.triage_act_time_dist_p3 = Lognormal(
            mean=self.param.mean_triage_time_dict[3],
            stdev=self.param.sd_triage_time_dict[3],
            random_seed=seeds[10]
        )
        # NEW
        self.triage_act_time_dist_p4 = Lognormal(
            mean=self.param.mean_triage_time_dict[4],
            stdev=self.param.sd_triage_time_dict[4],
            random_seed=seeds[11]
        )
        # NEW
        self.triage_act_time_dist_p5 = Lognormal(
            mean=self.param.mean_triage_time_dict[5],
            stdev=self.param.sd_triage_time_dict[5],
            random_seed=seeds[12]
        )
        # NEW
        self.treat_act_time_dist_p1 = Lognormal(
            mean=self.param.mean_treat_time_dict[1],
            stdev=self.param.sd_treat_time_dict[1],
            random_seed=seeds[3]
        )
        # NEW
        self.treat_act_time_dist_p2 = Lognormal(
            mean=self.param.mean_treat_time_dict[2],
            stdev=self.param.sd_treat_time_dict[2],
            random_seed=seeds[13]
        )
        # NEW
        self.treat_act_time_dist_p3 = Lognormal(
            mean=self.param.mean_treat_time_dict[3],
            stdev=self.param.sd_treat_time_dict[3],
            random_seed=seeds[14]
        )
        # NEW
        self.treat_act_time_dist_p4 = Lognormal(
            mean=self.param.mean_treat_time_dict[4],
            stdev=self.param.sd_treat_time_dict[4],
            random_seed=seeds[15]
        )
        # NEW
        self.treat_act_time_dist_p5 = Lognormal(
            mean=self.param.mean_treat_time_dict[5],
            stdev=self.param.sd_treat_time_dict[5],
            random_seed=seeds[16]
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
        # NEW
        self.patient_priority_rng = (
            np.random.default_rng(seeds[8])
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
        # NEW
        self.mean_q_time_doctor_pri_1 = pd.NA
        self.mean_q_time_doctor_pri_2 = pd.NA
        self.mean_q_time_doctor_pri_3 = pd.NA
        self.mean_q_time_doctor_pri_4 = pd.NA
        self.mean_q_time_doctor_pri_5 = pd.NA
        self.sd_q_time_doctor_pri_1 = pd.NA
        self.sd_q_time_doctor_pri_2 = pd.NA
        self.sd_q_time_doctor_pri_3 = pd.NA
        self.sd_q_time_doctor_pri_4 = pd.NA
        self.sd_q_time_doctor_pri_5 = pd.NA
        self.perc_90_q_time_doctor_pri_1 = pd.NA
        self.perc_90_q_time_doctor_pri_2 = pd.NA
        self.perc_90_q_time_doctor_pri_3 = pd.NA
        self.perc_90_q_time_doctor_pri_4 = pd.NA
        self.perc_90_q_time_doctor_pri_5 = pd.NA
        # NEW
        self.receptionist_util_total = 0
        self.receptionist_theo_unav_total = 0
        self.receptionist_util_prop = pd.NA
        self.nurse_util_total = 0
        self.nurse_theo_unav_total = 0
        self.nurse_util_prop = pd.NA
        self.doctor_util_total = 0
        self.doctor_theo_unav_total = 0
        self.doctor_util_prop = pd.NA
        self.pharmacist_util_total = 0
        self.pharmacist_theo_unav_total = 0
        self.pharmacist_util_prop = pd.NA

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1

            # NEW
            pat_pri_ran_gen = self.patient_priority_rng.random()

            if pat_pri_ran_gen < 0.05:
                patient_priority = 1
            elif pat_pri_ran_gen < 0.2:
                patient_priority = 2
            elif pat_pri_ran_gen < 0.6:
                patient_priority = 3
            elif pat_pri_ran_gen < 0.7:
                patient_priority = 4
            else:
                patient_priority = 5

            p = Patient(self.patient_counter, patient_priority) # NEW
            self.list_of_patients.append(p)
            self.env.process(self.attend_ed(p))
            sampled_inter = self.patient_inter_dist.sample(
                simulation_time=(self.env.now%1440)
            )
            yield self.env.timeout(sampled_inter)

    # NEW
    def obstruct_nurse(self):
        next_departure_time = self.param.nurse_unav_freq

        while True:
            yield self.env.timeout(
                next_departure_time - self.env.now
            )

            time_should_go = next_departure_time
            time_to_return = time_should_go + self.param.nurse_unav_time

            for removal_candidate in range(self.param.num_nurses_unav):
                self.env.process(
                    self.remove_one_nurse(
                        time_to_return
                    )
                )

            next_departure_time += (
                self.param.nurse_unav_time + self.param.nurse_unav_freq
            )

            if self.env.now > self.param.warm_up_period:
                for removal_candidate in range(self.param.num_nurses_unav):
                    if (time_to_return < self.param.sim_duration):
                        self.nurse_theo_unav_total += (
                            time_to_return - self.env.now
                        )
                    else:
                        self.nurse_theo_unav_total += (
                            self.param.sim_duration - self.env.now
                        )

    # NEW
    def obstruct_doctor(self):
        next_departure_time = self.param.doctor_unav_freq

        while True:
            yield self.env.timeout(
                next_departure_time - self.env.now
            )

            time_should_go = next_departure_time
            time_to_return = time_should_go + self.param.doctor_unav_time

            for removal_candidate in range(self.param.num_doctors_unav):
                self.env.process(
                    self.remove_one_doctor(
                        time_to_return
                    )
                )

            next_departure_time += (
                self.param.doctor_unav_time + self.param.doctor_unav_freq
            )

            if self.env.now > self.param.warm_up_period:
                for removal_candidate in range(self.param.num_doctors_unav):
                    if (time_to_return < self.param.sim_duration):
                        self.doctor_theo_unav_total += (
                            time_to_return - self.env.now
                        )
                    else:
                        self.doctor_theo_unav_total += (
                            self.param.sim_duration - self.env.now
                        )

    def remove_one_nurse(self, time_to_return):
        req = self.nurse.request(priority=-1)
        yield req
        time_went = self.env.now

        time_away = max(
            time_to_return - time_went, 0
        )
        yield self.env.timeout(time_away)
        self.nurse.release(req)

    def remove_one_doctor(self, time_to_return):
        req = self.doctor.request(priority=-1)
        yield req
        time_went = self.env.now

        time_away = max(
            time_to_return - time_went, 0
        )
        yield self.env.timeout(time_away)
        self.doctor.release(req)

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
        patient.arrival_time = self.env.now

        start_q_reg = self.env.now

        with self.receptionist.request() as req:
            yield req
            end_q_reg = self.env.now
            if self.env.now > self.param.warm_up_period:
                patient.q_time_reg = end_q_reg - start_q_reg
            sampled_reg_act_time = self.reg_act_time_dist.sample()
            
            # NEW
            if self.env.now > self.param.warm_up_period:
                end_activity = self.env.now + sampled_reg_act_time

                if (end_activity < self.param.sim_duration):
                    self.receptionist_util_total += sampled_reg_act_time
                else:
                    self.receptionist_util_total += (
                        self.param.sim_duration - self.env.now
                    )

            yield self.env.timeout(sampled_reg_act_time)

        start_q_triage = self.env.now
        
        with self.nurse.request() as req:
            yield req
            end_q_triage = self.env.now
            if self.env.now > self.param.warm_up_period:
                patient.q_time_triage = end_q_triage - start_q_triage
            # NEW
            if patient.priority == 1:
                chosen_triage_act_dist = self.triage_act_time_dist_p1
            elif patient.priority == 2:
                chosen_triage_act_dist = self.triage_act_time_dist_p2
            elif patient.priority == 3:
                chosen_triage_act_dist = self.triage_act_time_dist_p3
            elif patient.priority == 4:
                chosen_triage_act_dist = self.triage_act_time_dist_p4
            else:
                chosen_triage_act_dist = self.triage_act_time_dist_p5
            sampled_triage_act_time = chosen_triage_act_dist.sample()

            # NEW
            if self.env.now > self.param.warm_up_period:
                end_activity = self.env.now + sampled_triage_act_time

                if (end_activity < self.param.sim_duration):
                    self.nurse_util_total += sampled_triage_act_time
                else:
                    self.nurse_util_total += (
                        self.param.sim_duration - self.env.now
                    )

            yield self.env.timeout(sampled_triage_act_time)

        if (
            self.triage_pharm_branch_prob_rng.random() <
            self.param.branch_prob_triage_to_pharm_dict[patient.priority] # NEW
        ):
            start_q_pharm = self.env.now

            with self.pharmacist.request() as req:
                yield req
                end_q_pharm = self.env.now
                if self.env.now > self.param.warm_up_period:
                    patient.q_time_pharmacy = end_q_pharm - start_q_pharm
                sampled_pharm_act_time = self.pharm_act_time_dist.sample()
                
                # NEW
                if self.env.now > self.param.warm_up_period:
                    end_activity = self.env.now + sampled_pharm_act_time

                    if (end_activity < self.param.sim_duration):
                        self.pharmacist_util_total += sampled_pharm_act_time
                    else:
                        self.pharmacist_util_total += (
                            self.param.sim_duration - self.env.now
                        )

                yield self.env.timeout(sampled_pharm_act_time)

            # SINK AFTER PHARMACY
        else:
            start_q_treat = self.env.now

            with self.doctor.request(priority=patient.priority) as req: # NEW
                yield req
                end_q_treat = self.env.now

                if self.env.now > self.param.warm_up_period:
                    patient.q_time_treat = end_q_treat - start_q_treat
                # NEW
                if patient.priority == 1:
                    chosen_treat_act_dist = self.treat_act_time_dist_p1
                elif patient.priority == 2:
                    chosen_treat_act_dist = self.treat_act_time_dist_p2
                elif patient.priority == 3:
                    chosen_treat_act_dist = self.treat_act_time_dist_p3
                elif patient.priority == 4:
                    chosen_treat_act_dist = self.treat_act_time_dist_p4
                else:
                    chosen_treat_act_dist = self.treat_act_time_dist_p5
                sampled_treat_act_time = chosen_treat_act_dist.sample()

                # NEW
                if self.env.now > self.param.warm_up_period:
                    end_activity = self.env.now + sampled_treat_act_time

                    if (end_activity < self.param.sim_duration):
                        self.doctor_util_total += sampled_treat_act_time
                    else:
                        self.doctor_util_total += (
                            self.param.sim_duration - self.env.now
                        )

                yield self.env.timeout(sampled_treat_act_time)

            if (
                self.treat_pharm_branch_prob_rng.random() <
                self.param.branch_prob_treat_to_pharm_dict[patient.priority]#NEW
            ):
                start_q_pharm = self.env.now

                with self.pharmacist.request() as req:
                    yield req
                    end_q_pharm = self.env.now
                    if self.env.now > self.param.warm_up_period:
                        patient.q_time_pharmacy = end_q_pharm - start_q_pharm
                    sampled_pharm_act_time = self.pharm_act_time_dist.sample()

                    # NEW
                    if self.env.now > self.param.warm_up_period:
                        end_activity = self.env.now + sampled_pharm_act_time

                        if (end_activity < self.param.sim_duration):
                            self.pharmacist_util_total += sampled_pharm_act_time
                        else:
                            self.pharmacist_util_total += (
                                self.param.sim_duration - self.env.now
                            )

                    yield self.env.timeout(sampled_pharm_act_time)

                # SINK AFTER PHARMACY
            
    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.obstruct_nurse()) # NEW
        self.env.process(self.obstruct_doctor()) # NEW
        self.env.run(until=self.param.sim_duration)

    def run_warm_up_assessment(self):
        old_warm_up = self.param.warm_up_period
        self.param.warm_up_period = 0
        self.param.sim_duration_warm_up_assessment = (
            self.param.results_collection_period *
            self.param.warm_up_assessment_sim_length_scaler
        )
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.cumulative_mean_tracker())
        self.env.process(self.obstruct_nurse()) # NEW
        self.env.process(self.obstruct_doctor()) # NEW
        self.env.run(until=self.param.sim_duration_warm_up_assessment)
        self.param.warm_up_period = old_warm_up

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dataframe = pd.DataFrame(
            entity.__dict__ for entity in entity_list
        )

        return entity_dataframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_reg = entity_dataframe["q_time_reg"].mean()
        self.sd_q_time_reg = entity_dataframe["q_time_reg"].std()
        self.perc_90_q_time_reg = entity_dataframe["q_time_reg"].quantile(0.9)

        self.mean_q_time_triage = entity_dataframe["q_time_triage"].mean()
        self.sd_q_time_triage = entity_dataframe["q_time_triage"].std()
        self.perc_90_q_time_triage = (
            entity_dataframe["q_time_triage"].quantile(0.9)
        )

        self.mean_q_time_treat = entity_dataframe["q_time_treat"].mean()
        self.sd_q_time_treat = entity_dataframe["q_time_treat"].std()
        self.perc_90_q_time_treat = (
            entity_dataframe["q_time_treat"].quantile(0.9)
        )

        self.mean_q_time_pharm = entity_dataframe["q_time_pharmacy"].mean()
        self.sd_q_time_pharm = entity_dataframe["q_time_pharmacy"].std()
        self.perc_90_q_time_pharm = (
            entity_dataframe["q_time_pharmacy"].quantile(0.9)
        )

        self.replication_arrival_times = entity_dataframe["arrival_time"]

        # NEW
        self.mean_q_time_doctor_pri_1 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 1,
                "q_time_treat"
            ].mean()
        )
        self.mean_q_time_doctor_pri_2 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 2,
                "q_time_treat"
            ].mean()
        )
        self.mean_q_time_doctor_pri_3 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 3,
                "q_time_treat"
            ].mean()
        )
        self.mean_q_time_doctor_pri_4 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 4,
                "q_time_treat"
            ].mean()
        )
        self.mean_q_time_doctor_pri_5 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 5,
                "q_time_treat"
            ].mean()
        )
        self.sd_q_time_doctor_pri_1 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 1,
                "q_time_treat"
            ].std()
        )
        self.sd_q_time_doctor_pri_2 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 2,
                "q_time_treat"
            ].std()
        )
        self.sd_q_time_doctor_pri_3 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 3,
                "q_time_treat"
            ].std()
        )
        self.sd_q_time_doctor_pri_4 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 4,
                "q_time_treat"
            ].std()
        )
        self.sd_q_time_doctor_pri_5 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 5,
                "q_time_treat"
            ].std()
        )
        self.perc_90_q_time_doctor_pri_1 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 1,
                "q_time_treat"
            ].quantile(0.9)
        )
        self.perc_90_q_time_doctor_pri_2 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 2,
                "q_time_treat"
            ].quantile(0.9)
        )
        self.perc_90_q_time_doctor_pri_3 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 3,
                "q_time_treat"
            ].quantile(0.9)
        )
        self.perc_90_q_time_doctor_pri_4 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 4,
                "q_time_treat"
            ].quantile(0.9)
        )
        self.perc_90_q_time_doctor_pri_5 = (
            entity_dataframe.loc[
                entity_dataframe["priority"] == 5,
                "q_time_treat"
            ].quantile(0.9)
        )

        # NEW
        self.receptionist_util_prop = (
            self.receptionist_util_total / (
                (
                    self.param.results_collection_period * 
                    self.param.num_receptionists
                ) - self.receptionist_theo_unav_total
            )
        )

        # NEW
        self.nurse_util_prop = (
            self.nurse_util_total / (
                (
                    self.param.results_collection_period *
                    self.param.num_nurses
                ) - self.nurse_theo_unav_total
            )
        )

        # NEW
        self.doctor_util_prop = (
            self.doctor_util_total / (
                (
                    self.param.results_collection_period *
                    self.param.num_doctors
                ) - self.doctor_theo_unav_total
            )
        )

        # NEW
        self.pharmacist_util_prop = (
            self.pharmacist_util_total / (
                (
                    self.param.results_collection_period *
                    self.param.num_pharmacists
                ) - self.pharmacist_theo_unav_total
            )
        )

class Trial:
    def __init__(self, param, name_of_trial="Trial"):
        self.param = param
        self.name_of_trial = name_of_trial
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_reg = pd.NA
        self.trial_sd_q_time_reg = pd.NA
        self.trial_perc_90_q_time_reg = pd.NA
        self.trial_mean_q_time_triage = pd.NA
        self.trial_sd_q_time_triage = pd.NA
        self.trial_perc_90_q_time_triage = pd.NA
        self.trial_mean_q_time_treat = pd.NA
        self.trial_sd_q_time_treat = pd.NA
        self.trial_perc_90_q_time_treat = pd.NA
        self.trial_mean_q_time_pharm = pd.NA
        self.trial_sd_q_time_pharm = pd.NA
        self.trial_perc_90_q_time_pharm = pd.NA

        self.ci_lower_q_time_reg = pd.NA
        self.ci_upper_q_time_reg = pd.NA
        self.se_q_time_reg = pd.NA
        self.ci_lower_q_time_triage = pd.NA
        self.ci_upper_q_time_triage = pd.NA
        self.se_q_time_triage = pd.NA
        self.ci_lower_q_time_treat = pd.NA
        self.ci_upper_q_time_treat = pd.NA
        self.se_q_time_treat = pd.NA
        self.ci_lower_q_time_pharm = pd.NA
        self.ci_upper_q_time_pharm = pd.NA
        self.se_q_time_pharm = pd.NA
        self.warm_up_trial = False

        # NEW
        self.trial_mean_q_time_doctor_pri_1 = pd.NA
        self.trial_mean_q_time_doctor_pri_2 = pd.NA
        self.trial_mean_q_time_doctor_pri_3 = pd.NA
        self.trial_mean_q_time_doctor_pri_4 = pd.NA
        self.trial_mean_q_time_doctor_pri_5 = pd.NA
        self.trial_sd_q_time_doctor_pri_1 = pd.NA
        self.trial_sd_q_time_doctor_pri_2 = pd.NA
        self.trial_sd_q_time_doctor_pri_3 = pd.NA
        self.trial_sd_q_time_doctor_pri_4 = pd.NA
        self.trial_sd_q_time_doctor_pri_5 = pd.NA
        self.trial_perc_90_q_time_doctor_pri_1 = pd.NA
        self.trial_perc_90_q_time_doctor_pri_2 = pd.NA
        self.trial_perc_90_q_time_doctor_pri_3 = pd.NA
        self.trial_perc_90_q_time_doctor_pri_4 = pd.NA
        self.trial_perc_90_q_time_doctor_pri_5 = pd.NA
        self.se_q_time_doctor_pri_1 = pd.NA
        self.se_q_time_doctor_pri_2 = pd.NA
        self.se_q_time_doctor_pri_3 = pd.NA
        self.se_q_time_doctor_pri_4 = pd.NA
        self.se_q_time_doctor_pri_5 = pd.NA
        self.ci_lower_q_time_doctor_pri_1 = pd.NA
        self.ci_lower_q_time_doctor_pri_2 = pd.NA
        self.ci_lower_q_time_doctor_pri_3 = pd.NA
        self.ci_lower_q_time_doctor_pri_4 = pd.NA
        self.ci_lower_q_time_doctor_pri_5 = pd.NA
        self.ci_upper_q_time_doctor_pri_1 = pd.NA
        self.ci_upper_q_time_doctor_pri_2 = pd.NA
        self.ci_upper_q_time_doctor_pri_3 = pd.NA
        self.ci_upper_q_time_doctor_pri_4 = pd.NA
        self.ci_upper_q_time_doctor_pri_5 = pd.NA
        self.trial_mean_receptionist_util_prop = pd.NA
        self.trial_mean_nurse_util_prop = pd.NA
        self.trial_mean_doctor_util_prop = pd.NA
        self.trial_mean_pharmacist_util_prop = pd.NA

    def run_trial(self):
        for replication_id in tqdm(
            range(self.param.num_replications),
            desc=f"Running {self.name_of_trial}",
            unit="replication"
        ):
            model_replication = Model(self.param, replication_id)
            model_replication.run_model()
            patient_df = model_replication.convert_entity_list_to_dataframe(
                model_replication.list_of_patients
            )
            model_replication.calculate_run_results(patient_df)
            self.list_of_simulation_replications.append(model_replication)

    def run_warm_up_assessment_trial(self):
        self.warm_up_trial = True
        self.list_of_cumulative_mean_dfs = []

        for wu_replication_id in tqdm(
            range(self.param.num_replications_warm_up_assessment),
            desc=f"Running {self.name_of_trial}",
            unit="replication"
        ):
            wu_model_replication = Model(self.param, wu_replication_id)
            wu_model_replication.run_warm_up_assessment()
            patient_df = wu_model_replication.convert_entity_list_to_dataframe(
                wu_model_replication.list_of_patients
            )
            wu_model_replication.calculate_run_results(patient_df)
            self.list_of_simulation_replications.append(wu_model_replication)
            self.list_of_cumulative_mean_dfs.append(
                wu_model_replication.cumulative_mean_df
            )

        reference_df = self.list_of_cumulative_mean_dfs[0]
        x_col = "Simulation Time"
        y_cols = [
            col for col in reference_df.columns if col not in [
                x_col,
                "id",
                "arrival_time",
                "priority" # NEW
            ]
        ]

        for col in y_cols:
            fig = go.Figure()

            for i, df in enumerate(
                self.list_of_cumulative_mean_dfs, start=1
            ):
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode="lines",
                    name=f"replication_{i}",
                    line=dict(color="lightblue", width=1)
                ))

            combined = []

            for df in self.list_of_cumulative_mean_dfs:
                combined.append(df[[x_col, col]])

            df_all_reps = pd.concat(combined)

            df_all_reps = df_all_reps.sort_values(x_col)

            mean_across_reps_df = (
                df_all_reps.groupby(x_col, as_index=False)[col].mean()
            )

            mean_across_reps_df["overall_cumulative"] = (
                mean_across_reps_df[col].expanding().mean()
            )

            fig.add_trace(go.Scatter(
                x=mean_across_reps_df[x_col],
                y=mean_across_reps_df["overall_cumulative"],
                mode="lines",
                name="overall_mean",
                line=dict(color="darkblue", width=4)
            ))

            fig.update_layout(
                title=f"Cumulative Mean - {col}",
                xaxis_title=x_col,
                yaxis_title="Cumulative Mean"
            )

            fig.show()
            # NEW - renamed file to ed_model_3
            fig.write_html(f"ed_model_3_cumul_mean_{col}.html")

    def calculate_trial_results(self):
        if self.warm_up_trial:
            total_reps = self.param.num_replications_warm_up_assessment
        else:
            total_reps = self.param.num_replications

        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in
            self.list_of_simulation_replications
        )

        self.trial_mean_q_time_reg = (
            self.replication_df["mean_q_time_reg"].mean()
        )
        self.trial_sd_q_time_reg = (
            self.replication_df["mean_q_time_reg"].std()
        )
        self.trial_perc_90_q_time_reg = (
            self.replication_df["mean_q_time_reg"].quantile(0.9)
        )

        self.trial_mean_q_time_triage = (
            self.replication_df["mean_q_time_triage"].mean()
        )
        self.trial_sd_q_time_triage = (
            self.replication_df["mean_q_time_triage"].std()
        )
        self.trial_perc_90_q_time_triage = (
            self.replication_df["mean_q_time_triage"].quantile(0.9)
        )

        self.trial_mean_q_time_treat = (
            self.replication_df["mean_q_time_treat"].mean()
        )
        self.trial_sd_q_time_treat = (
            self.replication_df["mean_q_time_treat"].std()
        )
        self.trial_perc_90_q_time_treat = (
            self.replication_df["mean_q_time_treat"].quantile(0.9)
        )

        self.trial_mean_q_time_pharm = (
            self.replication_df["mean_q_time_pharm"].mean()
        )
        self.trial_sd_q_time_pharm = (
            self.replication_df["mean_q_time_pharm"].std()
        )
        self.trial_perc_90_q_time_pharm = (
            self.replication_df["mean_q_time_pharm"].quantile(0.9)
        )

        self.se_q_time_reg = (
            self.trial_sd_q_time_reg / math.sqrt(total_reps)
        )
        self.se_q_time_triage = (
            self.trial_sd_q_time_triage / math.sqrt(total_reps)
        )
        self.se_q_time_treat = (
            self.trial_sd_q_time_treat / math.sqrt(total_reps)
        )
        self.se_q_time_pharm = (
            self.trial_sd_q_time_pharm / math.sqrt(total_reps)
        )

        t = stats.t.ppf(0.975, df=total_reps-1)

        self.ci_lower_q_time_reg = (
            self.trial_mean_q_time_reg - (t * self.se_q_time_reg)
        )
        self.ci_upper_q_time_reg = (
            self.trial_mean_q_time_reg + (t * self.se_q_time_reg)
        )

        self.ci_lower_q_time_triage = (
            self.trial_mean_q_time_triage - (t * self.se_q_time_triage)
        )
        self.ci_upper_q_time_triage = (
            self.trial_mean_q_time_triage + (t * self.se_q_time_triage)
        )

        self.ci_lower_q_time_treat = (
            self.trial_mean_q_time_treat - (t * self.se_q_time_treat)
        )
        self.ci_upper_q_time_treat = (
            self.trial_mean_q_time_treat + (t * self.se_q_time_treat)
        )

        self.ci_lower_q_time_pharm = (
            self.trial_mean_q_time_pharm - (t * self.se_q_time_pharm)
        )
        self.ci_upper_q_time_pharm = (
            self.trial_mean_q_time_pharm + (t * self.se_q_time_pharm)
        )

        # NEW
        self.trial_mean_q_time_doctor_pri_1 = (
            self.replication_df["mean_q_time_doctor_pri_1"].mean()
        )
        self.trial_mean_q_time_doctor_pri_2 = (
            self.replication_df["mean_q_time_doctor_pri_2"].mean()
        )
        self.trial_mean_q_time_doctor_pri_3 = (
            self.replication_df["mean_q_time_doctor_pri_3"].mean()
        )
        self.trial_mean_q_time_doctor_pri_4 = (
            self.replication_df["mean_q_time_doctor_pri_4"].mean()
        )
        self.trial_mean_q_time_doctor_pri_5 = (
            self.replication_df["mean_q_time_doctor_pri_5"].mean()
        )
        self.trial_sd_q_time_doctor_pri_1 = (
            self.replication_df["mean_q_time_doctor_pri_1"].std()
        )
        self.trial_sd_q_time_doctor_pri_2 = (
            self.replication_df["mean_q_time_doctor_pri_2"].std()
        )
        self.trial_sd_q_time_doctor_pri_3 = (
            self.replication_df["mean_q_time_doctor_pri_3"].std()
        )
        self.trial_sd_q_time_doctor_pri_4 = (
            self.replication_df["mean_q_time_doctor_pri_4"].std()
        )
        self.trial_sd_q_time_doctor_pri_5 = (
            self.replication_df["mean_q_time_doctor_pri_5"].std()
        )
        self.trial_perc_90_q_time_doctor_pri_1 = (
            self.replication_df["mean_q_time_doctor_pri_1"].quantile(0.9)
        )
        self.trial_perc_90_q_time_doctor_pri_2 = (
            self.replication_df["mean_q_time_doctor_pri_2"].quantile(0.9)
        )
        self.trial_perc_90_q_time_doctor_pri_3 = (
            self.replication_df["mean_q_time_doctor_pri_3"].quantile(0.9)
        )
        self.trial_perc_90_q_time_doctor_pri_4 = (
            self.replication_df["mean_q_time_doctor_pri_4"].quantile(0.9)
        )
        self.trial_perc_90_q_time_doctor_pri_5 = (
            self.replication_df["mean_q_time_doctor_pri_5"].quantile(0.9)
        )
        self.se_q_time_doctor_pri_1 = (
            self.trial_sd_q_time_doctor_pri_1 / math.sqrt(total_reps)
        )
        self.se_q_time_doctor_pri_2 = (
            self.trial_sd_q_time_doctor_pri_2 / math.sqrt(total_reps)
        )
        self.se_q_time_doctor_pri_3 = (
            self.trial_sd_q_time_doctor_pri_3 / math.sqrt(total_reps)
        )
        self.se_q_time_doctor_pri_4 = (
            self.trial_sd_q_time_doctor_pri_4 / math.sqrt(total_reps)
        )
        self.se_q_time_doctor_pri_5 = (
            self.trial_sd_q_time_doctor_pri_5 / math.sqrt(total_reps)
        )
        self.ci_lower_q_time_doctor_pri_1 = (
            self.trial_mean_q_time_doctor_pri_1 - (
                t * self.se_q_time_doctor_pri_1
            )
        )
        self.ci_lower_q_time_doctor_pri_2 = (
            self.trial_mean_q_time_doctor_pri_2 - (
                t * self.se_q_time_doctor_pri_2
            )
        )
        self.ci_lower_q_time_doctor_pri_3 = (
            self.trial_mean_q_time_doctor_pri_3 - (
                t * self.se_q_time_doctor_pri_3
            )
        )
        self.ci_lower_q_time_doctor_pri_4 = (
            self.trial_mean_q_time_doctor_pri_4 - (
                t * self.se_q_time_doctor_pri_4
            )
        )
        self.ci_lower_q_time_doctor_pri_5 = (
            self.trial_mean_q_time_doctor_pri_5 - (
                t * self.se_q_time_doctor_pri_5
            )
        )
        self.ci_upper_q_time_doctor_pri_1 = (
            self.trial_mean_q_time_doctor_pri_1 + (
                t * self.se_q_time_doctor_pri_1
            )
        )
        self.ci_upper_q_time_doctor_pri_2 = (
            self.trial_mean_q_time_doctor_pri_2 + (
                t * self.se_q_time_doctor_pri_2
            )
        )
        self.ci_upper_q_time_doctor_pri_3 = (
            self.trial_mean_q_time_doctor_pri_3 + (
                t * self.se_q_time_doctor_pri_3
            )
        )
        self.ci_upper_q_time_doctor_pri_4 = (
            self.trial_mean_q_time_doctor_pri_4 + (
                t * self.se_q_time_doctor_pri_4
            )
        )
        self.ci_upper_q_time_doctor_pri_5 = (
            self.trial_mean_q_time_doctor_pri_5 + (
                t * self.se_q_time_doctor_pri_5
            )
        )

        # NEW
        self.trial_mean_receptionist_util_prop = (
            self.replication_df["receptionist_util_prop"].mean()
        )
        self.trial_mean_nurse_util_prop = (
            self.replication_df["nurse_util_prop"].mean()
        )
        self.trial_mean_doctor_util_prop = (
            self.replication_df["doctor_util_prop"].mean()
        )
        self.trial_mean_pharmacist_util_prop = (
            self.replication_df["pharmacist_util_prop"].mean()
        )

    def plot_arrival_time_frequencies(self):
        self.arrival_times_df = pd.DataFrame(
            columns=["arrival_time"]
        )
        
        for replication in self.list_of_simulation_replications:
            self.arrival_times_df = pd.concat(
                [
                    self.arrival_times_df,
                    replication.replication_arrival_times.to_frame()
                ],
                ignore_index=True
            )

        self.arrival_times_df['arr_time_bins'] = (
            pd.cut(
                self.arrival_times_df["arrival_time"],
                bins=[i for i in range(0, self.param.sim_duration+1, 60)],
                include_lowest=True,
                right=False
            )
        )

        self.arrival_times_df_grouped = (
            self.arrival_times_df
            .groupby("arr_time_bins")
            .count()
            .reset_index()
        )

        self.arrival_times_df_grouped["arr_times_bins_str"] = (
            self.arrival_times_df_grouped["arr_time_bins"].astype("str")
        )

        self.arrival_times_df_grouped["mean_arrivals_in_period_per_rep"] = (
            self.arrival_times_df_grouped["arrival_time"] / 
            self.param.num_replications
        )

        fig = px.line(
            self.arrival_times_df_grouped,
            x="arr_times_bins_str",
            y="mean_arrivals_in_period_per_rep"
        )

        fig.show()
        fig.write_html("ed_arrival_time_frequencies.html")

base_case_params = Param(
    patient_iat_csv="ed_iat_table.csv" # NEW - removed warm up override
)

#warm_up_assessment_trial = Trial(base_case_params, "Warm Up Assessment")
#warm_up_assessment_trial.run_warm_up_assessment_trial()
#warm_up_assessment_trial.calculate_trial_results()

list_of_trials = []

base_case_trial = Trial(base_case_params, "Base Case")
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()
base_case_trial.plot_arrival_time_frequencies()
list_of_trials.append(base_case_trial)

wi_1_params = Param(
    num_receptionists=1,
    num_nurses=5,
    num_pharmacists=1,
    patient_iat_csv="ed_iat_table.csv"
)

wi_1_trial = Trial(
    wi_1_params,
    "What If Scenario 1 : 1 Receptionist, 5 Nurses, 1 Pharmacist"
)
wi_1_trial.run_trial()
wi_1_trial.calculate_trial_results()
list_of_trials.append(wi_1_trial)

for trial in list_of_trials:
    print (trial.name_of_trial)
    print ("-----------------")
    print ("Registration")
    print (
        f"Mean: {trial.trial_mean_q_time_reg:.2f} |",
        f"SD: {trial.trial_sd_q_time_reg:.2f} |",
        f"90th Perc: {trial.trial_perc_90_q_time_reg:.2f} |",
        f"SE: {trial.se_q_time_reg:.2f} |",
        f"95% CI: ({trial.ci_lower_q_time_reg:.2f},",
        f"{trial.ci_upper_q_time_reg:.2f})"
    )
    print ()

    print ("Triage")
    print (
        f"Mean: {trial.trial_mean_q_time_triage:.2f} |",
        f"SD: {trial.trial_sd_q_time_triage:.2f} |",
        f"90th Perc: {trial.trial_perc_90_q_time_triage:.2f} |",
        f"SE: {trial.se_q_time_triage:.2f} |",
        f"95% CI: ({trial.ci_lower_q_time_triage:.2f},",
        f"{trial.ci_upper_q_time_triage:.2f})"
    )
    print ()

    print ("Treatment")
    print (
        f"Mean: {trial.trial_mean_q_time_treat:.2f} |",
        f"SD: {trial.trial_sd_q_time_treat:.2f} |",
        f"90th Perc: {trial.trial_perc_90_q_time_treat:.2f} |",
        f"SE: {trial.se_q_time_treat:.2f} |",
        f"95% CI: ({trial.ci_lower_q_time_treat:.2f},",
        f"{trial.ci_upper_q_time_treat:.2f})"
    )
    print ()

    # NEW
    print ("Treatment (by Priority)")
    print (
        "("
        "\033[31m"
        "P1,",
        "\033[0m"
        "\033[35m"
        "P2,",
        "\033[0m"
        "\033[33m"
        "P3,",
        "\033[0m"
        "\033[36m"
        "P4,",
        "\033[0m"
        "\033[32m"
        "P5)"
        "\033[0m"
    )
    print (
        "Mean : "
        "\033[31m"
        f"{trial.trial_mean_q_time_doctor_pri_1:.2f},",
        "\033[0m"
        "\033[35m"
        f"{trial.trial_mean_q_time_doctor_pri_2:.2f},",
        "\033[0m"
        "\033[33m"
        f"{trial.trial_mean_q_time_doctor_pri_3:.2f},",
        "\033[0m"
        "\033[36m"
        f"{trial.trial_mean_q_time_doctor_pri_4:.2f},",
        "\033[0m"
        "\033[32m"
        f"{trial.trial_mean_q_time_doctor_pri_5:.2f}",
        "\033[0m"
        "minutes"
    )
    print (
        "SD : "
        "\033[31m"
        f"{trial.trial_sd_q_time_doctor_pri_1:.2f},",
        "\033[0m"
        "\033[35m"
        f"{trial.trial_sd_q_time_doctor_pri_2:.2f},",
        "\033[0m"
        "\033[33m"
        f"{trial.trial_sd_q_time_doctor_pri_3:.2f},",
        "\033[0m"
        "\033[36m"
        f"{trial.trial_sd_q_time_doctor_pri_4:.2f},",
        "\033[0m"
        "\033[32m"
        f"{trial.trial_sd_q_time_doctor_pri_5:.2f}",
        "\033[0m"
        "minutes"
    )
    print (
        "90th Perc : "
        "\033[31m"
        f"{trial.trial_perc_90_q_time_doctor_pri_1:.2f},",
        "\033[0m"
        "\033[35m"
        f"{trial.trial_perc_90_q_time_doctor_pri_2:.2f},",
        "\033[0m"
        "\033[33m"
        f"{trial.trial_perc_90_q_time_doctor_pri_3:.2f},",
        "\033[0m"
        "\033[36m"
        f"{trial.trial_perc_90_q_time_doctor_pri_4:.2f},",
        "\033[0m"
        "\033[32m"
        f"{trial.trial_perc_90_q_time_doctor_pri_5:.2f}",
        "\033[0m"
        "minutes"
    )
    print (
        "SE : "
        "\033[31m"
        f"{trial.se_q_time_doctor_pri_1:.2f},",
        "\033[0m"
        "\033[35m"
        f"{trial.se_q_time_doctor_pri_2:.2f},",
        "\033[0m"
        "\033[33m"
        f"{trial.se_q_time_doctor_pri_3:.2f},",
        "\033[0m"
        "\033[36m"
        f"{trial.se_q_time_doctor_pri_4:.2f},",
        "\033[0m"
        "\033[32m"
        f"{trial.se_q_time_doctor_pri_5:.2f}",
        "\033[0m"
        "minutes"
    )
    print (
        "95% CI : "
        "\033[31m"
        f"({trial.ci_lower_q_time_doctor_pri_1:.2f},",
        f"{trial.ci_upper_q_time_doctor_pri_1:.2f}),"
        "\033[0m"
        "\033[35m"
        f"({trial.ci_lower_q_time_doctor_pri_2:.2f},",
        f"{trial.ci_upper_q_time_doctor_pri_2:.2f}),"
        "\033[0m"
        "\033[33m"
        f"({trial.ci_lower_q_time_doctor_pri_3:.2f},",
        f"{trial.ci_upper_q_time_doctor_pri_3:.2f}),"
        "\033[0m"
        "\033[36m"
        f"({trial.ci_lower_q_time_doctor_pri_4:.2f},",
        f"{trial.ci_upper_q_time_doctor_pri_4:.2f}),"
        "\033[0m"
        "\033[32m"
        f"({trial.ci_lower_q_time_doctor_pri_5:.2f},",
        f"{trial.ci_upper_q_time_doctor_pri_5:.2f}),"
        "\033[0m"
        "minutes"
    )

    print ()

    print ("Pharmacy")
    print (
        f"Mean: {trial.trial_mean_q_time_pharm:.2f} |",
        f"SD: {trial.trial_sd_q_time_pharm:.2f} |",
        f"90th Perc: {trial.trial_perc_90_q_time_pharm:.2f} |",
        f"SE: {trial.se_q_time_pharm:.2f} |",
        f"95% CI: ({trial.ci_lower_q_time_pharm:.2f},",
        f"{trial.ci_upper_q_time_pharm:.2f})"
    )
    print ()

    # NEW
    print ("MEAN RESOURCE UTILISATION")
    print (
        "Receptionist : ",
        f"{trial.trial_mean_receptionist_util_prop*100:.2f}%"
    )

    print (
        "Nurse : ",
        f"{trial.trial_mean_nurse_util_prop*100:.2f}%"
    )

    print (
        "Doctor : ",
        f"{trial.trial_mean_doctor_util_prop*100:.2f}%"
    )

    print (
        "Pharmacist : ",
        f"{trial.trial_mean_pharmacist_util_prop*100:.2f}%"
    )


import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px # NEW
import plotly.graph_objects as go # NEW

class Patient:
    def __init__(self, p_id):
        self.id = p_id

        self.q_time_nurse = pd.NA

class Param:
    def __init__(
        self,
        mean_patient_inter = 7,
        mean_nurse_consult_time = 6,
        sd_nurse_consult_time = 1,
        num_nurses = 1,
        results_collection_period = 120,
        warm_up_period = 60,
        num_replications = 100,
        num_replications_warm_up_assessment = 5, # NEW
        warm_up_asessment_sim_length_scaler = 20, # NEW
        cumulative_mean_tracker_interval = 5 # NEW
    ):
        self.mean_patient_inter = mean_patient_inter
        self.mean_nurse_consult_time = mean_nurse_consult_time
        self.sd_nurse_consult_time = sd_nurse_consult_time
        self.num_nurses = num_nurses
        self.results_collection_period = results_collection_period
        self.warm_up_period = warm_up_period
        self.sim_duration = warm_up_period + results_collection_period
        self.num_replications = num_replications
        # NEW
        self.num_replications_warm_up_assessment = (
            num_replications_warm_up_assessment
        )
        # NEW
        self.sim_duration_warm_up_assessment = (
            self.sim_duration * warm_up_asessment_sim_length_scaler
        )
        # NEW
        self.cumulative_mean_tracker_interval = (
            cumulative_mean_tracker_interval
        )

class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.nurse = simpy.Resource(self.env, capacity=self.param.num_nurses)

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(2)
        self.patient_inter_dist = Exponential(
            mean=self.param.mean_patient_inter,
            random_seed=seeds[0]
        )
        self.nurse_consult_time_dist = Lognormal(
            mean=self.param.mean_nurse_consult_time,
            stdev=self.param.sd_nurse_consult_time,
            random_seed=seeds[1]
        )

        self.list_of_patients = []
        self.mean_q_time_nurse = pd.NA
        self.sd_q_time_nurse = pd.NA
        self.perc_90_q_time_nurse = pd.NA

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.list_of_patients.append(p)
            self.env.process(self.attend_clinic(p))
            sampled_inter = self.patient_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    # NEW
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

    def attend_clinic(self, patient):
        start_q_nurse = self.env.now

        with self.nurse.request() as req:
            yield req
            end_q_nurse = self.env.now
            if self.env.now > self.param.warm_up_period:
                patient.q_time_nurse = end_q_nurse - start_q_nurse
            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)

    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.run(until=self.param.sim_duration)

    # NEW
    def run_warm_up_assessment(self):
        self.param.num_replications = (
            self.param.num_replications_warm_up_assessment
        )
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.cumulative_mean_tracker())
        self.env.run(until=self.param.sim_duration_warm_up_assessment)

        print (self.cumulative_mean_df) # NEW TEMP REMOVE

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dateframe = pd.DataFrame(
            entity.__dict__ for entity in entity_list
        )

        return entity_dateframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_nurse = (
            entity_dataframe["q_time_nurse"].mean()
        )
        self.sd_q_time_nurse = entity_dataframe["q_time_nurse"].std()
        self.perc_90_q_time_nurse = (
            entity_dataframe["q_time_nurse"].quantile(0.9)
        )

class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_nurse = pd.NA
        self.trial_sd_q_time_nurse = pd.NA
        self.trial_perc_90_q_time_nurse = pd.NA
        self.ci_lower_q_time_nurse = pd.NA
        self.ci_upper_q_time_nurse = pd.NA
        self.se_q_time_nurse = pd.NA
    
    def run_trial(self):
        for replication_id in range(self.param.num_replications):
            model_replication = Model(self.param, replication_id)
            model_replication.run_model()
            patient_df = model_replication.convert_entity_list_to_dataframe(
                model_replication.list_of_patients
            )
            model_replication.calculate_run_results(patient_df)
            self.list_of_simulation_replications.append(model_replication)

    # NEW
    def run_warm_up_assessment_trial(self):
        self.list_of_cumulative_mean_dfs = []

        for wu_replication_id in range(
            self.param.num_replications_warm_up_assessment
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

        x_col = "Simulation Time"
        y_cols = [
            col for col in df_filtered.columns if col not in [x_col, "id"]
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
                    name=f"replication_{i}"
                ))

            fig.update_layout(
                title=f"Cumulative Mean - {col}",
                xaxis_title=x_col,
                yaxis_title="Cumulative Mean"
            )

            fig.show()
            fig.write_html(f"cumul_mean_{col}.html")



        for i, df in enumerate(
            self.list_of_cumulative_mean_dfs, start=1
        ):
            df_filtered = df.drop(
                columns=[
                    "id"
                ]
            )
            
            x_col = "Simulation Time"
            y_cols = [col for col in df_filtered.columns if col != x_col]

            fig = px.line(
                df_filtered,
                x=x_col,
                y=y_cols,
                labels={
                    "value":"Cumulative Mean"
                },
            )
            fig.show()
            fig.write_html(f"cumul_mean_rep_{i}.html")

        # HERE!!!! Converting to running as a trial, before plotting results
        # Don't forget to change code at bottom too

    def calculate_trial_results(self):
        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in 
            self.list_of_simulation_replications
        )

        self.trial_mean_q_time_nurse = (
            self.replication_df["mean_q_time_nurse"].mean()
        )

        self.trial_sd_q_time_nurse = (
            self.replication_df["mean_q_time_nurse"].std()
        )

        self.trial_perc_90_q_time_nurse = (
            self.replication_df["mean_q_time_nurse"].quantile(0.9)
        )

        self.se_q_time_nurse = (
            self.trial_sd_q_time_nurse / math.sqrt(self.param.num_replications)
        )

        t = stats.t.ppf(0.975, df=self.param.num_replications-1)

        self.ci_lower_q_time_nurse = (
            self.trial_mean_q_time_nurse - (t * self.se_q_time_nurse)
        )

        self.ci_upper_q_time_nurse = (
            self.trial_mean_q_time_nurse + (t * self.se_q_time_nurse)
        )

base_case_params = Param()

# NEW
warm_up_assessment_trial = Trial(base_case_params)
warm_up_assessment_trial.run_warm_up_assessment_trial()
warm_up_assessment_trial.calculate_trial_results()

base_case_trial = Trial(base_case_params)
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()
print ("BASE CASE TRIAL RESULTS")
print ("-----------------------")
print ("Queuing Time for the Nurse")
print (f"Mean : {base_case_trial.trial_mean_q_time_nurse:.2f} minutes")
print (f"SD : {base_case_trial.trial_sd_q_time_nurse:.2f} minutes")
print (f"90th Perc : {base_case_trial.trial_perc_90_q_time_nurse:.2f} minutes")
print (f"Standard Error : {base_case_trial.se_q_time_nurse:.2f}")
print (
    f"95% CI : ({base_case_trial.ci_lower_q_time_nurse:.2f}, ",
    f"{base_case_trial.ci_upper_q_time_nurse:.2f}) minutes"
)
print ()

what_if_params = Param(num_nurses=2)
what_if_trial = Trial(what_if_params)
what_if_trial.run_trial()
what_if_trial.calculate_trial_results()
print ("2 NURSES TRIAL RESULTS")
print ("----------------------")
print ("Queuing Time for the Nurse")
print (f"Mean : {what_if_trial.trial_mean_q_time_nurse:.2f} minutes")
print (f"SD : {what_if_trial.trial_sd_q_time_nurse:.2f} minutes")
print (f"90th Perc : {what_if_trial.trial_perc_90_q_time_nurse:.2f} minutes")
print (f"Standard Error : {what_if_trial.se_q_time_nurse:.2f}")
print (
    f"95% CI : ({what_if_trial.ci_lower_q_time_nurse:.2f}, ",
    f"{what_if_trial.ci_upper_q_time_nurse:.2f}) minutes"
)
print ()


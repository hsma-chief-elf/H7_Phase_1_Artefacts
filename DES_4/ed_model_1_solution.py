import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
import math
from scipy import stats
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm

class Patient:
    def __init__(self, p_id):
        self.id = p_id

        self.q_time_reg = pd.NA
        self.q_time_triage = pd.NA
        self.q_time_treat = pd.NA
        self.q_time_pharmacy = pd.NA

class Param:
    def __init__(
        self,
        mean_patient_inter = 6,
        mean_reg_time = 4,
        sd_reg_time = 1.2,
        mean_triage_time = 10,
        sd_triage_time = 2,
        mean_treat_time = 45,
        sd_treat_time = 20,
        mean_pharm_time = 6,
        sd_pharm_time = 2,
        num_receptionists = 1,
        num_nurses = 2,
        num_doctors = 5,
        num_pharmacists = 1,
        branch_prob_triage_to_pharm = 0.5,
        branch_prob_treat_to_pharm = 0.4,
        results_collection_period = 2880,
        warm_up_period = 10000,
        num_replications = 100,
        num_replications_warm_up_assessment = 20,
        warm_up_assessment_sim_length_scaler = 10,
        cumulative_mean_tracker_interval = 100
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
            
    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.run(until=self.param.sim_duration)

    def run_warm_up_assessment(self):
        self.param.warm_up_period = 0
        self.param.sim_duration_warm_up_assessment = (
            self.param.results_collection_period *
            self.param.warm_up_assessment_sim_length_scaler
        )
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.cumulative_mean_tracker())
        self.env.run(until=self.param.sim_duration_warm_up_assessment)

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
            col for col in reference_df.columns if col not in [x_col, "id"]
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
            fig.write_html(f"ed_model_1_cumul_mean_{col}.html")

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

base_case_params = Param()

warm_up_assessment_trial = Trial(base_case_params, "Warm Up Assessment")
warm_up_assessment_trial.run_warm_up_assessment_trial()
warm_up_assessment_trial.calculate_trial_results()

base_case_trial = Trial(base_case_params, "Base Case")
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()
print ({base_case_trial.name_of_trial})
print ("-----------------")
print ("Registration")
print (
    f"Mean: {base_case_trial.trial_mean_q_time_reg:.2f} |",
    f"SD: {base_case_trial.trial_sd_q_time_reg:.2f} |",
    f"90th Perc: {base_case_trial.trial_perc_90_q_time_reg:.2f} |",
    f"SE: {base_case_trial.se_q_time_reg:.2f} |",
    f"95% CI: ({base_case_trial.ci_lower_q_time_reg:.2f},",
    f"{base_case_trial.ci_upper_q_time_reg:.2f})"
)
print ()

print ("Triage")
print (
    f"Mean: {base_case_trial.trial_mean_q_time_triage:.2f} |",
    f"SD: {base_case_trial.trial_sd_q_time_triage:.2f} |",
    f"90th Perc: {base_case_trial.trial_perc_90_q_time_triage:.2f} |",
    f"SE: {base_case_trial.se_q_time_triage:.2f} |",
    f"95% CI: ({base_case_trial.ci_lower_q_time_triage:.2f},",
    f"{base_case_trial.ci_upper_q_time_triage:.2f})"
)
print ()

print ("Treatment")
print (
    f"Mean: {base_case_trial.trial_mean_q_time_treat:.2f} |",
    f"SD: {base_case_trial.trial_sd_q_time_treat:.2f} |",
    f"90th Perc: {base_case_trial.trial_perc_90_q_time_treat:.2f} |",
    f"SE: {base_case_trial.se_q_time_treat:.2f} |",
    f"95% CI: ({base_case_trial.ci_lower_q_time_treat:.2f},",
    f"{base_case_trial.ci_upper_q_time_treat:.2f})"
)
print ()

print ("Pharmacy")
print (
    f"Mean: {base_case_trial.trial_mean_q_time_pharm:.2f} |",
    f"SD: {base_case_trial.trial_sd_q_time_pharm:.2f} |",
    f"90th Perc: {base_case_trial.trial_perc_90_q_time_pharm:.2f} |",
    f"SE: {base_case_trial.se_q_time_pharm:.2f} |",
    f"95% CI: ({base_case_trial.ci_lower_q_time_pharm:.2f},",
    f"{base_case_trial.ci_upper_q_time_pharm:.2f})"
)
print ()

wi_1_params = Param(mean_patient_inter=3)

wi_1_trial = Trial(
    wi_1_params,
    "What If Scenario 1 : Doubled Arrivals, No Resource Change"
)
wi_1_trial.run_trial()
wi_1_trial.calculate_trial_results()
print (wi_1_trial.name_of_trial)
print ("-----------------")
print ("Registration")
print (
    f"Mean: {wi_1_trial.trial_mean_q_time_reg:.2f} |",
    f"SD: {wi_1_trial.trial_sd_q_time_reg:.2f} |",
    f"90th Perc: {wi_1_trial.trial_perc_90_q_time_reg:.2f} |",
    f"SE: {wi_1_trial.se_q_time_reg:.2f} |",
    f"95% CI: ({wi_1_trial.ci_lower_q_time_reg:.2f},",
    f"{wi_1_trial.ci_upper_q_time_reg:.2f})"
)
print ()

print ("Triage")
print (
    f"Mean: {wi_1_trial.trial_mean_q_time_triage:.2f} |",
    f"SD: {wi_1_trial.trial_sd_q_time_triage:.2f} |",
    f"90th Perc: {wi_1_trial.trial_perc_90_q_time_triage:.2f} |",
    f"SE: {wi_1_trial.se_q_time_triage:.2f} |",
    f"95% CI: ({wi_1_trial.ci_lower_q_time_triage:.2f},",
    f"{wi_1_trial.ci_upper_q_time_triage:.2f})"
)
print ()

print ("Treatment")
print (
    f"Mean: {wi_1_trial.trial_mean_q_time_treat:.2f} |",
    f"SD: {wi_1_trial.trial_sd_q_time_treat:.2f} |",
    f"90th Perc: {wi_1_trial.trial_perc_90_q_time_treat:.2f} |",
    f"SE: {wi_1_trial.se_q_time_treat:.2f} |",
    f"95% CI: ({wi_1_trial.ci_lower_q_time_treat:.2f},",
    f"{wi_1_trial.ci_upper_q_time_treat:.2f})"
)
print ()

print ("Pharmacy")
print (
    f"Mean: {wi_1_trial.trial_mean_q_time_pharm:.2f} |",
    f"SD: {wi_1_trial.trial_sd_q_time_pharm:.2f} |",
    f"90th Perc: {wi_1_trial.trial_perc_90_q_time_pharm:.2f} |",
    f"SE: {wi_1_trial.se_q_time_pharm:.2f} |",
    f"95% CI: ({wi_1_trial.ci_lower_q_time_pharm:.2f},",
    f"{wi_1_trial.ci_upper_q_time_pharm:.2f})"
)
print ()


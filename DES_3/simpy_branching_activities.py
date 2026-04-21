import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
import math
from scipy import stats
import numpy as np

class Patient:
    def __init__(self, p_id):
        self.id = p_id

        self.q_time_registration = pd.NA
        self.q_time_nurse = pd.NA
        self.q_time_specialist = pd.NA # NEW

class Param:
    def __init__(
        self,
        mean_patient_inter = 5,
        mean_registration_time = 3,
        sd_registration_time = 0.5,
        mean_nurse_consult_time = 6,
        sd_nurse_consult_time = 1,
        mean_specialist_time = 60, # NEW
        sd_specialist_time = 5, # NEW
        num_receptionists = 1,
        num_nurses = 1,
        num_specialists = 1, # NEW
        specialist_prob = 0.3, # NEW
        sim_duration = 120,
        num_replications = 5
    ):
        self.mean_patient_inter = mean_patient_inter
        self.mean_registration_time = mean_registration_time
        self.sd_registration_time = sd_registration_time
        self.mean_nurse_consult_time = mean_nurse_consult_time
        self.sd_nurse_consult_time = sd_nurse_consult_time
        self.mean_specialist_time = mean_specialist_time # NEW
        self.sd_specialist_time = sd_specialist_time # NEW
        self.num_receptionists = num_receptionists
        self.num_nurses = num_nurses
        self.num_specialists = num_specialists # NEW
        self.specialist_prob = specialist_prob # NEW
        self.sim_duration = sim_duration
        self.num_replications = num_replications

class Model:
    def __init__(self, param, replication_id):
        self.param = param
        self.replication_id = replication_id
        self.env = simpy.Environment()
        self.patient_counter = 0

        self.receptionist = simpy.Resource(
            self.env, capacity=self.param.num_receptionists
        )
        self.nurse = simpy.Resource(self.env, capacity=self.param.num_nurses)
        # NEW
        self.specialist = simpy.Resource(
            self.env, capacity=self.param.num_specialists
        )

        ss = np.random.SeedSequence(self.replication_id)
        seeds = ss.spawn(5) # NEW
        self.patient_inter_dist = Exponential(
            mean=self.param.mean_patient_inter,
            random_seed=seeds[0]
        )
        self.registration_time_dist = Lognormal(
            mean=self.param.mean_registration_time,
            stdev=self.param.sd_registration_time,
            random_seed=seeds[2]
        )
        self.nurse_consult_time_dist = Lognormal(
            mean=self.param.mean_nurse_consult_time,
            stdev=self.param.sd_nurse_consult_time,
            random_seed=seeds[1]
        )
        # NEW
        self.specialist_branch_prob_rng = (
            np.random.default_rng(seeds[3])
        )
        # NEW
        self.specialist_time_dist = Lognormal(
            mean=self.param.mean_specialist_time,
            stdev=self.param.sd_specialist_time,
            random_seed=seeds[4]
        )

        self.list_of_patients = []
        self.mean_q_time_registration = pd.NA
        self.sd_q_time_registration = pd.NA
        self.perc_90_q_time_registration = pd.NA
        self.mean_q_time_nurse = pd.NA
        self.sd_q_time_nurse = pd.NA
        self.perc_90_q_time_nurse = pd.NA
        self.mean_q_time_specialist = pd.NA # NEW
        self.sd_q_time_specialist = pd.NA # NEW
        self.perc_90_q_time_specialist = pd.NA # NEW

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.list_of_patients.append(p)
            self.env.process(self.attend_clinic(p))
            sampled_inter = self.patient_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    def attend_clinic(self, patient):
        start_q_registration = self.env.now

        with self.receptionist.request() as req:
            yield req
            end_q_registration = self.env.now
            patient.q_time_registration = (
                end_q_registration - start_q_registration
            )
            sampled_reg_act_time = self.registration_time_dist.sample()
            yield self.env.timeout(sampled_reg_act_time)

        start_q_nurse = self.env.now

        with self.nurse.request() as req:
            yield req
            end_q_nurse = self.env.now
            patient.q_time_nurse = end_q_nurse - start_q_nurse
            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)

        # NEW
        if (
            self.specialist_branch_prob_rng.random() < 
            self.param.specialist_prob
        ):
            start_q_specialist = self.env.now

            with self.specialist.request() as req:
                yield req
                end_q_specialist = self.env.now
                patient.q_time_specialist = (
                    end_q_specialist - start_q_specialist
                )
                sampled_specialist_act_time = self.specialist_time_dist.sample()
                yield self.env.timeout(sampled_specialist_act_time)

    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.run(until=self.param.sim_duration)

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dateframe = pd.DataFrame(
            entity.__dict__ for entity in entity_list
        )

        return entity_dateframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_registration = (
            entity_dataframe["q_time_registration"].mean()
        )
        self.sd_q_time_registration = (
            entity_dataframe["q_time_registration"].std()
        )
        self.perc_90_q_time_registration = (
            entity_dataframe["q_time_registration"].quantile(0.9)
        )

        self.mean_q_time_nurse = (
            entity_dataframe["q_time_nurse"].mean()
        )
        self.sd_q_time_nurse = entity_dataframe["q_time_nurse"].std()
        self.perc_90_q_time_nurse = (
            entity_dataframe["q_time_nurse"].quantile(0.9)
        )

        # NEW
        self.mean_q_time_specialist = (
            entity_dataframe["q_time_specialist"].mean()
        )
        self.sd_q_time_specialist = (
            entity_dataframe["q_time_specialist"].std()
        )
        self.perc_90_q_time_specialist = (
            entity_dataframe["q_time_specialist"].quantile(0.9)
        )

class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_registration = pd.NA
        self.trial_sd_q_time_registration = pd.NA
        self.trial_perc_90_q_time_registration = pd.NA
        self.trial_mean_q_time_nurse = pd.NA
        self.trial_sd_q_time_nurse = pd.NA
        self.trial_perc_90_q_time_nurse = pd.NA
        self.trial_mean_q_time_specialist = pd.NA # NEW
        self.trial_sd_q_time_specialist = pd.NA # NEW
        self.trial_perc_90_q_time_specialist = pd.NA # NEW
        self.ci_lower_q_time_registration = pd.NA
        self.ci_upper_q_time_registration = pd.NA
        self.se_q_time_registration = pd.NA
        self.ci_lower_q_time_nurse = pd.NA
        self.ci_upper_q_time_nurse = pd.NA
        self.se_q_time_nurse = pd.NA
        self.ci_lower_q_time_specialist = pd.NA # NEW
        self.ci_upper_q_time_specialist = pd.NA # NEW
        self.se_q_time_specialist = pd.NA # NEW
    
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

        self.trial_mean_q_time_registration = (
            self.replication_df["mean_q_time_registration"].mean()
        )
        self.trial_sd_q_time_registration = (
            self.replication_df["mean_q_time_registration"].std()
        )
        self.trial_perc_90_q_time_registration = (
            self.replication_df["mean_q_time_registration"].quantile(0.9)
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

        # NEW
        self.trial_mean_q_time_specialist = (
            self.replication_df["mean_q_time_specialist"].mean()
        )
        self.trial_sd_q_time_specialist = (
            self.replication_df["mean_q_time_specialist"].std()
        )
        self.trial_perc_90_q_time_specialist = (
            self.replication_df["mean_q_time_specialist"].quantile(0.9)
        )

        self.se_q_time_registration = (
            self.trial_sd_q_time_registration / math.sqrt(
                self.param.num_replications
            )
        )
        self.se_q_time_nurse = (
            self.trial_sd_q_time_nurse / math.sqrt(self.param.num_replications)
        )
        # NEW
        self.se_q_time_specialist = (
            self.trial_sd_q_time_specialist / math.sqrt(
                self.param.num_replications
            )
        )

        t = stats.t.ppf(0.975, df=self.param.num_replications-1)

        self.ci_lower_q_time_registration = (
            self.trial_mean_q_time_registration - (
                t * self.se_q_time_registration
            )
        )
        self.ci_upper_q_time_registration = (
            self.trial_mean_q_time_registration + (
                t * self.se_q_time_registration
            )
        )
        
        self.ci_lower_q_time_nurse = (
            self.trial_mean_q_time_nurse - (t * self.se_q_time_nurse)
        )
        self.ci_upper_q_time_nurse = (
            self.trial_mean_q_time_nurse + (t * self.se_q_time_nurse)
        )

        # NEW
        self.ci_lower_q_time_specialist = (
            self.trial_mean_q_time_specialist - (t * self.se_q_time_specialist)
        )
        self.ci_upper_q_time_specialist = (
            self.trial_mean_q_time_specialist + (t * self.se_q_time_specialist)
        )

base_case_params = Param()
base_case_trial = Trial(base_case_params)
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()
print ("BASE CASE TRIAL RESULTS")
print ("-----------------------")
print ("Queuing Time for Registration")
print (f"Mean : {base_case_trial.trial_mean_q_time_registration:.2f} minutes")
print (f"SD : {base_case_trial.trial_sd_q_time_registration:.2f} minutes")
print (
    f"90th Perc : {base_case_trial.trial_perc_90_q_time_registration:.2f}",
    "minutes"
)
print (f"Standard Error : {base_case_trial.se_q_time_registration:.2f}")
print (
    f"95% CI : ({base_case_trial.ci_lower_q_time_registration:.2f}, ",
    f"{base_case_trial.ci_upper_q_time_registration:.2f}) minutes"
)
print ()

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

# NEW
print ("Queuing Time for the Specialist")
print (f"Mean : {base_case_trial.trial_mean_q_time_specialist:.2f} minutes")
print (f"SD : {base_case_trial.trial_sd_q_time_specialist:.2f} minutes")
print (
    f"90th Perc : {base_case_trial.trial_perc_90_q_time_specialist:.2f} ",
    "minutes"
)
print (f"Standard Error : {base_case_trial.se_q_time_specialist:.2f}")
print (
    f"95% CI : ({base_case_trial.ci_lower_q_time_specialist:.2f}, ",
    f"{base_case_trial.ci_upper_q_time_specialist:.2f}) minutes"
)
print ()

what_if_params = Param(num_nurses=2, num_receptionists=2)
what_if_trial = Trial(what_if_params)
what_if_trial.run_trial()
what_if_trial.calculate_trial_results()
print ("2 NURSES, 2 RECEPTIONISTS, 1 SPECIALIST TRIAL RESULTS")
print ("----------------------")
print ("Queuing Time for Registration")
print (f"Mean : {what_if_trial.trial_mean_q_time_registration:.2f} minutes")
print (f"SD : {what_if_trial.trial_sd_q_time_registration:.2f} minutes")
print (
    f"90th Perc : {what_if_trial.trial_perc_90_q_time_registration:.2f}",
    "minutes"
)
print (f"Standard Error : {what_if_trial.se_q_time_registration:.2f}")
print (
    f"95% CI : ({what_if_trial.ci_lower_q_time_registration:.2f}, ",
    f"{what_if_trial.ci_upper_q_time_registration:.2f}) minutes"
)
print ()

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

# NEW
print ("Queuing Time for the Specialist")
print (f"Mean : {what_if_trial.trial_mean_q_time_specialist:.2f} minutes")
print (f"SD : {what_if_trial.trial_sd_q_time_specialist:.2f} minutes")
print (
    f"90th Perc : {what_if_trial.trial_perc_90_q_time_specialist:.2f} ",
    "minutes"
)
print (f"Standard Error : {what_if_trial.se_q_time_specialist:.2f}")
print (
    f"95% CI : ({what_if_trial.ci_lower_q_time_specialist:.2f}, ",
    f"{what_if_trial.ci_upper_q_time_specialist:.2f}) minutes"
)
print ()


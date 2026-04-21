import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd

class Patient:
    def __init__(self, p_id):
        self.id = p_id

        self.q_time_nurse = pd.NA

class Param:
    def __init__(
        self,
        mean_patient_inter = 5,
        mean_nurse_consult_time = 6,
        sd_nurse_consult_time = 1,
        num_nurses = 1,
        sim_duration = 120,
        num_replications = 5
    ):
        self.mean_patient_inter = mean_patient_inter
        self.mean_nurse_consult_time = mean_nurse_consult_time
        self.sd_nurse_consult_time = sd_nurse_consult_time
        self.num_nurses = num_nurses
        self.sim_duration = sim_duration
        self.num_replications = num_replications

class Model:
    def __init__(self, param):
        self.param = param
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.nurse = simpy.Resource(self.env, capacity=self.param.num_nurses)
        self.patient_inter_dist = Exponential(
            mean=self.param.mean_patient_inter
        )
        self.nurse_consult_time_dist = Lognormal(
            mean=self.param.mean_nurse_consult_time,
            stdev=self.param.sd_nurse_consult_time
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

    def attend_clinic(self, patient):
        start_q_nurse = self.env.now

        with self.nurse.request() as req:
            yield req
            end_q_nurse = self.env.now
            patient.q_time_nurse = end_q_nurse - start_q_nurse
            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)

    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.run(until=self.param.sim_duration)

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

# NEW
class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_nurse = pd.NA
        self.trial_sd_q_time_nurse = pd.NA
        self.trial_perc_90_q_time_nurse = pd.NA
    
    def run_trial(self):
        for replication_id in range(self.param.num_replications):
            model_replication = Model(self.param)
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
base_case_params = Param()
base_case_trial = Trial(base_case_params)
base_case_trial.run_trial()
base_case_trial.calculate_trial_results()
print ("BASE CASE TRIAL RESULTS")
print ("-----------------------")
print ("Queuing Time for the Nurse")
print (f"Mean : {base_case_trial.trial_mean_q_time_nurse:.2f} minutes")
print (f"SD : {base_case_trial.trial_sd_q_time_nurse:.2f} minutes")
print (f"90th Perc : {base_case_trial.trial_perc_90_q_time_nurse:.2f} minutes")
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
print ()


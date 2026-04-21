import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
import random
import math
from scipy import stats

class HSMA:
    def __init__(self, id):
        self.id = id
        self.py_confidence = random.choice(["Weak", "Normal", "Strong"])
        self.q_time_trainer = pd.NA

class Param:
    def __init__(
        self,
        mean_hsma_inter = 20,
        mean_trainer_time_weak = 60,
        mean_trainer_time_normal = 30,
        mean_trainer_time_strong = 10,
        sd_trainer_time_weak = 10,
        sd_trainer_time_normal = 8,
        sd_trainer_time_strong = 2,
        num_trainers = 2,
        sim_duration = 480,
        num_replications = 5
    ):
        self.mean_hsma_inter = mean_hsma_inter
        self.mean_trainer_time_weak = mean_trainer_time_weak
        self.mean_trainer_time_normal = mean_trainer_time_normal
        self.mean_trainer_time_strong = mean_trainer_time_strong
        self.sd_trainer_time_weak = sd_trainer_time_weak
        self.sd_trainer_time_normal = sd_trainer_time_normal
        self.sd_trainer_time_strong = sd_trainer_time_strong
        self.num_trainers = num_trainers
        self.sim_duration = sim_duration
        self.num_replications = num_replications

class Model:
    def __init__(self, param):
        self.param = param
        self.env = simpy.Environment()
        self.hsma_counter = 0
        self.trainer = simpy.Resource(
            self.env, capacity=self.param.num_trainers
        )

        self.hsma_inter_dist = Exponential(
            mean=self.param.mean_hsma_inter
        )
        self.trainer_time_weak_dist = Lognormal(
            mean=self.param.mean_trainer_time_weak,
            stdev=self.param.sd_trainer_time_weak
        )
        self.trainer_time_normal_dist = Lognormal(
            mean=self.param.mean_trainer_time_normal,
            stdev=self.param.sd_trainer_time_normal
        )
        self.trainer_time_strong_dist = Lognormal(
            mean=self.param.mean_trainer_time_strong,
            stdev=self.param.sd_trainer_time_strong
        )

        self.list_of_hsmas = []
        self.mean_q_time_trainer = pd.NA
        self.sd_q_time_trainer = pd.NA
        self.perc_95_q_time_trainer = pd.NA
        self.mean_q_time_trainer_weak = pd.NA
        self.sd_q_time_trainer_weak = pd.NA
        self.perc_95_q_time_trainer_weak = pd.NA

    def generator_hsma_arrivals(self):
        while True:
            self.hsma_counter += 1
            associate = HSMA(self.hsma_counter)
            self.list_of_hsmas.append(associate)
            self.env.process(self.attend_consultation(associate))
            sampled_inter = self.hsma_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    def attend_consultation(self, associate):
        start_q_trainer = self.env.now

        with self.trainer.request() as req:
            yield req
            end_q_trainer = self.env.now
            associate.q_time_trainer = end_q_trainer - start_q_trainer
            
            if associate.py_confidence == "Weak":
                sampled_trainer_time = self.trainer_time_weak_dist.sample()
            elif associate.py_confidence == "Normal":
                sampled_trainer_time = self.trainer_time_normal_dist.sample()
            else:
                sampled_trainer_time = self.trainer_time_strong_dist.sample()

            yield self.env.timeout(sampled_trainer_time)

    def run_model(self):
        self.env.process(self.generator_hsma_arrivals())
        self.env.run(until=self.param.sim_duration)

    def convert_entity_list_to_dataframe(self, entity_list):
        entity_dataframe = pd.DataFrame(
            entity.__dict__ for entity in entity_list
        )

        return entity_dataframe

    def calculate_run_results(self, entity_dataframe):
        self.mean_q_time_trainer = (
            entity_dataframe["q_time_trainer"].mean()
        )
        self.sd_q_time_trainer = (
            entity_dataframe["q_time_trainer"].std()
        )
        self.perc_95_q_time_trainer = (
            entity_dataframe["q_time_trainer"].quantile(0.95)
        )

        self.mean_q_time_trainer_weak = (
            entity_dataframe.loc[
                entity_dataframe["py_confidence"] == "Weak",
                "q_time_trainer"
            ].mean()
        )

        self.sd_q_time_trainer_weak = (
            entity_dataframe.loc[
                entity_dataframe["py_confidence"] == "Weak",
                "q_time_trainer"
            ].std()
        )

        self.perc_95_q_time_trainer_weak = (
            entity_dataframe.loc[
                entity_dataframe["py_confidence"] == "Weak",
                "q_time_trainer"
            ].quantile(0.95)
        )

class Trial:
    def __init__(self, param):
        self.param = param
        self.list_of_simulation_replications = []
        self.trial_mean_q_time_trainer = pd.NA
        self.trial_sd_q_time_trainer = pd.NA
        self.trial_perc_95_q_time_trainer = pd.NA
        self.trial_mean_q_time_trainer_weak = pd.NA
        self.trial_sd_q_time_trainer_weak = pd.NA
        self.trial_perc_95_q_time_trainer_weak = pd.NA
        self.ci_lower_q_time_trainer = pd.NA
        self.ci_upper_q_time_trainer = pd.NA
        self.se_q_time_trainer = pd.NA
        self.ci_lower_q_time_trainer_weak = pd.NA
        self.ci_upper_q_time_trainer_weak = pd.NA
        self.se_q_time_trainer_weak = pd.NA

    def run_trial(self):
        for replication_id in range(self.param.num_replications):
            model_replication = Model(self.param)
            model_replication.run_model()
            hsma_df = model_replication.convert_entity_list_to_dataframe(
                model_replication.list_of_hsmas
            )
            model_replication.calculate_run_results(hsma_df)
            self.list_of_simulation_replications.append(model_replication)

    def calculate_trial_results(self):
        self.replication_df = pd.DataFrame(
            replication.__dict__ for replication in
            self.list_of_simulation_replications
        )

        self.trial_mean_q_time_trainer = (
            self.replication_df["mean_q_time_trainer"].mean()
        )

        self.trial_sd_q_time_trainer = (
            self.replication_df["mean_q_time_trainer"].std()
        )

        self.trial_perc_95_q_time_trainer = (
            self.replication_df["mean_q_time_trainer"].quantile(0.95)
        )

        self.trial_mean_q_time_trainer_weak = (
            self.replication_df["mean_q_time_trainer_weak"].mean()
        )

        self.trial_sd_q_time_trainer_weak = (
            self.replication_df["mean_q_time_trainer_weak"].std()
        )

        self.trial_perc_95_q_time_trainer_weak = (
            self.replication_df["mean_q_time_trainer_weak"].quantile(0.95)
        )

        self.se_q_time_trainer = (
            self.trial_sd_q_time_trainer / 
            math.sqrt(self.param.num_replications)
        )

        self.se_q_time_trainer_weak = (
            self.trial_sd_q_time_trainer_weak /
            math.sqrt(self.param.num_replications)
        )

        t = stats.t.ppf(0.975, df=self.param.num_replications-1)

        self.ci_lower_q_time_trainer = (
            self.trial_mean_q_time_trainer - (t * self.se_q_time_trainer)
        )

        self.ci_upper_q_time_trainer = (
            self.trial_mean_q_time_trainer + (t * self.se_q_time_trainer)
        )

        self.ci_lower_q_time_trainer_weak = (
            self.trial_mean_q_time_trainer_weak - (
                t * self.se_q_time_trainer_weak
            )
        )

        self.ci_upper_q_time_trainer_weak = (
            self.trial_mean_q_time_trainer_weak + (
                t * self.se_q_time_trainer_weak
            )
        )

base_params = Param()
base_trial = Trial(base_params)
base_trial.run_trial()
base_trial.calculate_trial_results()

print ("BASE CASE TRIAL RESULTS")
print ("-----------------------")
print ("Queuing Time for the Trainer (all HSMAs)")
print (f"Mean : {base_trial.trial_mean_q_time_trainer:.2f} minutes")
print (f"SD : {base_trial.trial_sd_q_time_trainer:.2f} minutes")
print (f"95th Perc : {base_trial.trial_perc_95_q_time_trainer:.2f} minutes")
print (f"Standard Error : {base_trial.se_q_time_trainer:.2f}")
print (
    f"95% CI : ({base_trial.ci_lower_q_time_trainer:.2f}, ",
    f"{base_trial.ci_upper_q_time_trainer:.2f}) minutes"
)
print ("Queuing Time for the Trainer (weak confidence HSMAs)")
print (f"Mean : {base_trial.trial_mean_q_time_trainer_weak:.2f} minutes")
print (f"SD : {base_trial.trial_sd_q_time_trainer_weak:.2f} minutes")
print (
    f"95th Perc : {base_trial.trial_perc_95_q_time_trainer_weak:.2f}",
    "minutes"
)
print (f"Standard Error : {base_trial.se_q_time_trainer_weak:.2f}")
print (
    f"95% CI : ({base_trial.ci_lower_q_time_trainer_weak:.2f}, ",
    f"{base_trial.ci_upper_q_time_trainer_weak:.2f}) minutes"
)
print ()

add_trainer_params = Param(num_trainers=3)
add_trainer_trial = Trial(add_trainer_params)
add_trainer_trial.run_trial()
add_trainer_trial.calculate_trial_results()

print ("3 TRAINER TRIAL RESULTS")
print ("-----------------------")
print ("Queuing Time for the Trainer (all HSMAs)")
print (f"Mean : {add_trainer_trial.trial_mean_q_time_trainer:.2f} minutes")
print (f"SD : {add_trainer_trial.trial_sd_q_time_trainer:.2f} minutes")
print (
    f"95th Perc : {add_trainer_trial.trial_perc_95_q_time_trainer:.2f} ",
    "minutes"
)
print (f"Standard Error : {add_trainer_trial.se_q_time_trainer:.2f}")
print (
    f"95% CI : ({add_trainer_trial.ci_lower_q_time_trainer:.2f}, ",
    f"{add_trainer_trial.ci_upper_q_time_trainer:.2f}) minutes"
)
print ("Queuing Time for the Trainer (weak confidence HSMAs)")
print (f"Mean : {add_trainer_trial.trial_mean_q_time_trainer_weak:.2f} minutes")
print (f"SD : {add_trainer_trial.trial_sd_q_time_trainer_weak:.2f} minutes")
print (
    f"95th Perc : {add_trainer_trial.trial_perc_95_q_time_trainer_weak:.2f} ",
    "minutes"
)
print (f"Standard Error : {add_trainer_trial.se_q_time_trainer_weak:.2f}")
print (
    f"95% CI : ({add_trainer_trial.ci_lower_q_time_trainer_weak:.2f}, ",
    f"{add_trainer_trial.ci_upper_q_time_trainer_weak:.2f}) minutes"
)
print ()


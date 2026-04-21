import simpy
from sim_tools.distributions import Exponential, Lognormal
import pandas as pd
import random

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

base_params = Param()
base_model = Model(base_params)
base_model.run_model()

base_hsma_df = base_model.convert_entity_list_to_dataframe(
    base_model.list_of_hsmas
)
base_model.calculate_run_results(base_hsma_df)
print ("Base Case Scenario")
print ("ALL HSMA TRAINER QUEUING TIMES")
print (f"Mean      : {base_model.mean_q_time_trainer:.2f}")
print (f"SD        : {base_model.sd_q_time_trainer:.2f}")
print (f"95th Perc : {base_model.perc_95_q_time_trainer:.2f}")
print ("WEAK CONFIDENCE HSMA TRAINER QUEUING TIMES")
print (f"Mean      : {base_model.mean_q_time_trainer_weak:.2f}")
print (f"SD        : {base_model.sd_q_time_trainer_weak:.2f}")
print (f"95th Perc : {base_model.perc_95_q_time_trainer_weak:.2f}")

add_trainer_params = Param(num_trainers=3)
add_trainer_model = Model(add_trainer_params)
add_trainer_model.run_model()

add_trainer_hsma_df = add_trainer_model.convert_entity_list_to_dataframe(
    add_trainer_model.list_of_hsmas
)
add_trainer_model.calculate_run_results(add_trainer_hsma_df)
print ("Additional Trainer Scenario")
print ("ALL HSMA TRAINER QUEUING TIMES")
print (f"Mean      : {add_trainer_model.mean_q_time_trainer:.2f}")
print (f"SD        : {add_trainer_model.sd_q_time_trainer:.2f}")
print (f"95th Perc : {add_trainer_model.perc_95_q_time_trainer:.2f}")
print ("WEAK CONFIDENCE HSMA TRAINER QUEUING TIMES")
print (f"Mean      : {add_trainer_model.mean_q_time_trainer_weak:.2f}")
print (f"SD        : {add_trainer_model.sd_q_time_trainer_weak:.2f}")
print (f"95th Perc : {add_trainer_model.perc_95_q_time_trainer_weak:.2f}")


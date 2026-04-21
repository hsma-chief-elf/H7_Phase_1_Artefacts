import simpy
from sim_tools.distributions import Exponential, Lognormal

class Patient:
    def __init__(self, p_id):
        self.id = p_id

        self.q_time_nurse = 0

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

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            print (f"Patient {p.id} has entered the clinic at {self.env.now}")
            self.env.process(self.attend_clinic(p))
            sampled_inter = self.patient_inter_dist.sample()
            yield self.env.timeout(sampled_inter)

    def attend_clinic(self, patient):
        start_q_nurse = self.env.now
        print (
            f"Patient {patient.id} starts queuing for the nurse at",
            f"{self.env.now}"
        )

        with self.nurse.request() as req:
            yield req
            end_q_nurse = self.env.now
            print (f"Patient {patient.id} sees the nurse at {self.env.now}")
            patient.q_time_nurse = end_q_nurse - start_q_nurse
            print (
                f"Patient {patient.id} waited {patient.q_time_nurse}",
                "minutes to see the nurse"
            )
            sampled_nurse_act_time = self.nurse_consult_time_dist.sample()
            yield self.env.timeout(sampled_nurse_act_time)
            print (
                f"Patient {patient.id} spent {sampled_nurse_act_time}",
                "minutes with the nurse"
            )

    def run_model(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.run(until=self.param.sim_duration)

print ("DEFAULT PARAMETERS")
print ("------------------")
my_params = Param()
my_model = Model(my_params)
my_model.run_model()

print ("ADJUSTED PARAMETERS")
print ("-------------------")
my_adjusted_params = Param(
    mean_patient_inter=2.5,
    mean_nurse_consult_time=12,
    sd_nurse_consult_time=2,
    num_nurses=2,
)
my_adjusted_model = Model(my_adjusted_params)
my_adjusted_model.run_model()


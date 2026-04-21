import streamlit as st
import simpy
import random
import Lognormal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Global parameters container
# ----------------------------
class g:
    patient_inter = 5
    mean_specialist_consult_time = 5
    sd_specialist_consult_time = 2
    number_of_specialists = 1
    sim_duration = 840


# ----------------------------
# Patient entity
# ----------------------------
class Patient:
    def __init__(self, p_id):
        self.id = p_id
        self.q_time_specialist = 0


# ----------------------------
# Model
# ----------------------------
class Model:
    def __init__(self, run_number, variable_times=False):
        self.env = simpy.Environment()
        self.run_number = run_number
        self.variable_times = variable_times
        self.patient_counter = 0

        self.specialist = simpy.PriorityResource(
            self.env, capacity=g.number_of_specialists
        )

        self.results_df = pd.DataFrame(
            {"Patient ID": [1], "Q Time Specialist": [0.0]}
        ).set_index("Patient ID")

        self.mean_q_time_specialist = 0
        self.dans_q_time = None

    def generator_patient_arrivals(self):
        while True:
            self.patient_counter += 1
            p = Patient(self.patient_counter)
            self.env.process(self.attend_clinic(p))

            if self.variable_times:
                sampled_inter = random.expovariate(1.0 / g.patient_inter)
                yield self.env.timeout(sampled_inter)
            else:
                yield self.env.timeout(g.patient_inter)

    def generator_dans_arrival(self):
        yield self.env.timeout(300)
        d = Patient("Dan")
        self.env.process(self.attend_clinic(d))

    def attend_clinic(self, patient):
        start_q = self.env.now
        with self.specialist.request() as req:
            yield req
            end_q = self.env.now
            patient.q_time_specialist = end_q - start_q
            self.results_df.loc[patient.id] = patient.q_time_specialist

            if self.variable_times:
                sampled_specialist_act_time = Lognormal.Lognormal(
                g.mean_specialist_consult_time,
                g.sd_specialist_consult_time).sample()

                act_time = Lognormal.Lognormal(
                    g.mean_specialist_consult_time,
                    g.sd_specialist_consult_time
                    ).sample()
            else:
                act_time = g.mean_specialist_consult_time

            yield self.env.timeout(act_time)

    def run(self):
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.generator_dans_arrival())
        self.env.run(until=g.sim_duration)

        self.results_df = self.results_df.drop(index=1, errors="ignore")
        self.mean_q_time_specialist = self.results_df["Q Time Specialist"].mean()
        if "Dan" in self.results_df.index:
            self.dans_q_time = self.results_df.loc["Dan"]["Q Time Specialist"]


# ----------------------------
# Trial
# ----------------------------
class Trial:
    def __init__(self, n_runs, variable_times=False):
        self.n_runs = n_runs
        self.variable_times = variable_times
        self.results = []

    def run(self):
        for r in range(self.n_runs):
            model = Model(r, self.variable_times)
            model.run()
            self.results.append(model.dans_q_time)

        return pd.Series(self.results, name="Dan Wait Time")


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Dan's Waiting Time Simulation", layout="wide")
st.title("🏥 How long does Dan wait?")

with st.sidebar:
    st.header("Simulation settings")
    variable = st.radio(
        "Consultation & arrival times",
        ["Fixed (deterministic)", "Variable (stochastic)"],
    )

    runs = st.slider("Number of runs", 50, 1000, 500, step=50)

    run_button = st.button("Run simulation")


if run_button:
    variable_flag = variable.startswith("Variable")

    with st.spinner("Running simulation..."):
        trial = Trial(runs, variable_flag)
        dan_waits = trial.run().dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Summary statistics")
        st.metric("Mean wait (min)", f"{dan_waits.mean():.1f}")
        st.metric("Median wait (min)", f"{dan_waits.median():.1f}")
        st.metric("90th percentile (min)", f"{dan_waits.quantile(0.9):.1f}")

    with col2:
        st.subheader("Distribution of Dan's waiting time")
        if dan_waits.nunique() == 1:
            st.info(
                "In the fixed (deterministic) case, Dan's waiting time is the same in every run, "
                "so a histogram is not meaningful."
            )
            st.metric("Dan always waits (min)", f"{dan_waits.iloc[0]:.1f}")
        else:
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = np.arange(0, dan_waits.max() + 10, 10)
            ax.hist(dan_waits, bins=bins, edgecolor="black")
            ax.set_xlabel("Wait time (minutes)")
            ax.set_ylabel("Frequency")
            ax.set_title("Histogram of Dan's wait")
            st.pyplot(fig)

    st.subheader("Why variability matters")
    st.write(
        "Even though average arrival and service times are the same, "
        "introducing randomness leads to longer queues and much worse tail behaviour."
    )

    st.dataframe(dan_waits.describe().round(2))

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Clean dataset (exact means)
# ----------------------------

# Inter-arrival times (mean = 5)
inter_arrivals = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 5], dtype=float)

# Arrival times
arrival_times = np.concatenate([[0], np.cumsum(inter_arrivals[:-1])])

# Service times (mean = 5)
service_times = np.array([3.2, 4.1, 4.5, 5.0, 5.4, 5.9, 6.1, 4.8, 3.7, 7.3])

patients = np.arange(1, len(service_times) + 1)

# ----------------------------
# Compute service start times
# ----------------------------

service_starts = []
current_time = 0.0

for arrival, service in zip(arrival_times, service_times):
    start = max(arrival, current_time)
    service_starts.append(start)
    current_time = start + service

service_starts = np.array(service_starts)

# ----------------------------
# Plot Gantt chart
# ----------------------------

plt.figure(figsize=(10, 5))

# Service bars
plt.barh(
    patients,
    service_times,
    left=service_starts
)

# Arrival ticks and waiting lines
for y, arrival, start in zip(patients, arrival_times, service_starts):
    # Arrival tick
    plt.plot([arrival, arrival], [y - 0.35, y + 0.35])
    # Waiting line (if any)
    if start > arrival:
        plt.plot([arrival, start], [y, y])

plt.xlabel("Time (minutes)")
plt.ylabel("Patient")
plt.title("Wellbeing Clinic Timeline with Arrival and Activity Times")
plt.yticks(patients)
plt.tight_layout()
plt.show()

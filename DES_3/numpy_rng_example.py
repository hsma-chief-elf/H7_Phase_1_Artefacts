import numpy as np

ss = np.random.SeedSequence(42)
seeds = ss.spawn(1)

rng = np.random.default_rng(seeds[0])

list_of_computers = [
    "Amiga 500",
    "Amiga 500+",
    "Amiga 600",
    "Amiga 1000",
    "Amiga 1200"
]

best_computer_of_all_time = rng.choice(list_of_computers)

print (best_computer_of_all_time)


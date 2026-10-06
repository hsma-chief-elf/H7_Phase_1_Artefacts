from tqdm import tqdm
import numpy as np

for i in tqdm(range(1000)):
    sum_of_array = np.random.rand(10_000_000).sum()


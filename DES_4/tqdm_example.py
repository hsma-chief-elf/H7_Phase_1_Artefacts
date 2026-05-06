from tqdm import tqdm

for i in tqdm(
    range(100000),
    desc="Running tqdm Example",
    unit="pointless calculation"
):
    answer = sum(j * j for j in range(5000))


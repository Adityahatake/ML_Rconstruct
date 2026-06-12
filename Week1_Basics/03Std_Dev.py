import numpy as np

a = [48, 50, 51, 49, 52]   # Student A
b = [10, 90, 20, 80, 50]   # Student B

print("Student A → Mean:", np.mean(a), "| Std Dev:", np.std(a))
print("Student B → Mean:", np.mean(b), "| Std Dev:", np.std(b))

# Outlier detection using 3-sigma rule
data = [10, 12, 11, 13, 10, 12, 95, 11]  # 95 is outlier

mean = np.mean(data)
std  = np.std(data)

for val in data:
    if abs(val - mean) > 3 * std:
        print(f"{val} is an OUTLIER")
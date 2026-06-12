import numpy as np
from scipy import stats

salaries = [30000, 35000, 32000, 28000, 200000]

print("Mean   :", np.mean(salaries))
print("Median :", np.median(salaries))
print("Mode   :", stats.mode(salaries).mode)  

# Now see the difference clearly
print("\nMean vs Median difference:", np.mean(salaries) - np.median(salaries))
# Large difference = outlier is pulling the mean
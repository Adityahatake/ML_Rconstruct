import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Right skewed data (like income)
income = [30, 32, 35, 33, 31, 34, 36, 38, 200, 500, 1000]

print("Mean   :", round(np.mean(income), 2))
print("Median :", np.median(income))
print("Skewness:", round(stats.skew(income), 2))
print("Kurtosis:", round(stats.kurtosis(income), 2))

# Visualize
plt.hist(income, bins=8, color='steelblue', edgecolor='black')
plt.axvline(np.mean(income), color='red', label='Mean')
plt.axvline(np.median(income), color='green', label='Median')
plt.legend()
plt.title("Right Skewed Distribution")
plt.show()
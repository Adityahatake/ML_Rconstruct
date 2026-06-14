import numpy as np
import matplotlib.pyplot as plt

salaries = [2, 4, 6, 8, 10, 12, 14, 16, 18, 200]

Q1 = np.percentile(salaries, 25)
Q3 = np.percentile(salaries, 75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Lower fence: {lower}, Upper fence: {upper}")

outliers = [x for x in salaries if x < lower or x > upper]
print(f"Outliers: {outliers}")

# Boxplot
plt.boxplot(salaries, vert=False)
plt.title("Salary Boxplot")
plt.show()
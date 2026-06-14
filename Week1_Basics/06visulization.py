import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy import stats 

# salary dataset[Fake]
np.random.seed(42)
salaries =np.concatenate([
np.random.normal(50000, 10000 , 95) , #95 normal employees
[200000, 250000, 300000, 400000, 500000] #5 executives high salary employees
])

df=pd.DataFrame({"salary": salaries})

# Descriptive stats
print("="*40)
print("Descriptive Statistics")
print("="*40)
print(f"Mean: {df['salary'].mean():.0f}")
print(f"Median: {df['salary'].median():.0f}")
print(f"Mode: {df['salary'].mode()[0]:.0f}")    
print(f"Standard Deviation: {df['salary'].std():.0f}")
print(f"Skewness: {stats.skew(df['salary']):.2f}")

Q1=np.percentile(salaries, 25)
Q3=np.percentile(salaries, 75)
IQR=Q3-Q1

print(f"Q1          : {Q1:.0f}")
print(f"Q3          : {Q3:.0f}")
print(f"IQR         : {IQR:.0f}")

upper_fence=Q3+1.5*IQR
lower_fence=Q1-1.5*IQR
outliers = salaries[salaries > upper_fence] #since salaries can not be negative, we only check for upper outliers
print(f"Outliers     : {outliers}")

#----- plots ------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(salaries, bins=20, color='steelblue', edgecolor='black')
axes[0].axvline(np.mean(salaries), color='red', linewidth=2, label='Mean')
axes[0].axvline(np.median(salaries), color='green', linewidth=2, label='Median')
axes[0].set_title('Salary Distribution (Histogram)')
axes[0].set_xlabel('Salary')
axes[0].legend()

# Boxplot
axes[1].boxplot(salaries, vert=False)
axes[1].set_title('Salary Distribution (Boxplot)')
axes[1].set_xlabel('Salary')

plt.tight_layout()
plt.show()
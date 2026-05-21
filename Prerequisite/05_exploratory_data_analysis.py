"""
Exploratory Data Analysis (EDA)
================================
EDA helps you understand your data deeply before building models.
Visualization is your friend! A picture is worth 1000 words.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# CREATE SAMPLE DATASET
# ============================================================================

data = {
    'Age': np.random.randint(22, 60, 100),
    'Salary': np.random.randint(40000, 150000, 100),
    'Experience': np.random.randint(0, 35, 100),
    'Department': np.random.choice(['Sales', 'IT', 'HR', 'Finance'], 100),
    'Performance_Score': np.random.randint(60, 100, 100)
}

df = pd.DataFrame(data)
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)
print("\nDataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())


# ============================================================================
# 1. UNIVARIATE ANALYSIS - NUMERIC VARIABLES
# ============================================================================

print("\n" + "="*70)
print("1. UNIVARIATE ANALYSIS - NUMERIC VARIABLES")
print("="*70)

numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Count: {df[col].count()}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std Dev: {df[col].std():.2f}")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Q1 (25%): {df[col].quantile(0.25):.2f}")
    print(f"  Q3 (75%): {df[col].quantile(0.75):.2f}")
    print(f"  Skewness: {df[col].skew():.2f}")
    print(f"  Kurtosis: {df[col].kurtosis():.2f}")


# ============================================================================
# 2. UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*70)
print("2. UNIVARIATE ANALYSIS - CATEGORICAL VARIABLES")
print("="*70)

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Value counts:")
    print(df[col].value_counts())
    print(f"  Percentages:")
    print((df[col].value_counts(normalize=True) * 100).round(2))


# ============================================================================
# 3. BIVARIATE ANALYSIS - NUMERIC vs NUMERIC
# ============================================================================

print("\n" + "="*70)
print("3. BIVARIATE ANALYSIS - NUMERIC vs NUMERIC")
print("="*70)

print("\nCorrelation Matrix:")
correlation_matrix = df[numeric_cols].corr()
print(correlation_matrix)

print("\nStrongest Correlations with Salary:")
salary_corr = correlation_matrix['Salary'].sort_values(ascending=False)
print(salary_corr)


# ============================================================================
# 4. BIVARIATE ANALYSIS - CATEGORICAL vs NUMERIC
# ============================================================================

print("\n" + "="*70)
print("4. BIVARIATE ANALYSIS - CATEGORICAL vs NUMERIC")
print("="*70)

print("\nAverage Salary by Department:")
dept_salary = df.groupby('Department')['Salary'].agg(['mean', 'median', 'min', 'max', 'count'])
print(dept_salary)

print("\nAverage Age by Department:")
print(df.groupby('Department')['Age'].describe())


# ============================================================================
# 5. DISTRIBUTION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("5. DISTRIBUTION ANALYSIS")
print("="*70)

print("\nSalary Distribution Statistics:")
print(f"  Mean: {df['Salary'].mean():.0f}")
print(f"  Median: {df['Salary'].median():.0f}")
print(f"  Skewness: {df['Salary'].skew():.4f}")

if df['Salary'].skew() > 0:
    print("  → Right-skewed (tail on right)")
elif df['Salary'].skew() < 0:
    print("  → Left-skewed (tail on left)")
else:
    print("  → Symmetric distribution")

print("\nAge Distribution (Histogram bins):")
bins = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60])
print(bins.value_counts().sort_index())


# ============================================================================
# 6. CREATING VISUALIZATIONS
# ============================================================================

print("\n" + "="*70)
print("6. CREATING VISUALIZATIONS")
print("="*70)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Histogram - Salary Distribution
axes[0, 0].hist(df['Salary'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Salary')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Salary Distribution')
axes[0, 0].axvline(df['Salary'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# Scatter plot - Age vs Salary
axes[0, 1].scatter(df['Age'], df['Salary'], alpha=0.6)
axes[0, 1].set_xlabel('Age')
axes[0, 1].set_ylabel('Salary')
axes[0, 1].set_title('Age vs Salary')

# Box plot - Salary by Department
df.boxplot(column='Salary', by='Department', ax=axes[1, 0])
axes[1, 0].set_xlabel('Department')
axes[1, 0].set_ylabel('Salary')
axes[1, 0].set_title('Salary Distribution by Department')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Bar plot - Department counts
dept_counts = df['Department'].value_counts()
axes[1, 1].bar(dept_counts.index, dept_counts.values, edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Department')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Number of Employees by Department')
plt.sca(axes[1, 1])
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\eda_visualizations.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualizations saved to 'eda_visualizations.png'")
plt.close()


# ============================================================================
# 7. OUTLIER DETECTION
# ============================================================================

print("\n" + "="*70)
print("7. OUTLIER DETECTION")
print("="*70)

def find_outliers_iqr(data, column):
    """
    Find outliers using Interquartile Range (IQR) method
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    print(f"\nOutlier Detection for {column}:")
    print(f"  Q1: {Q1:.0f}, Q3: {Q3:.0f}, IQR: {IQR:.0f}")
    print(f"  Lower bound: {lower_bound:.0f}, Upper bound: {upper_bound:.0f}")
    print(f"  Number of outliers: {len(outliers)}")
    
    return outliers

outliers_salary = find_outliers_iqr(df, 'Salary')
outliers_age = find_outliers_iqr(df, 'Age')


# ============================================================================
# 8. MULTIVARIATE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("8. MULTIVARIATE ANALYSIS")
print("="*70)

# Cross-tabulation
print("\nCross-tabulation: Department vs Performance Categories:")
df['Performance_Category'] = pd.cut(df['Performance_Score'], 
                                    bins=[0, 70, 85, 100], 
                                    labels=['Low', 'Medium', 'High'])

cross_tab = pd.crosstab(df['Department'], df['Performance_Category'])
print(cross_tab)

print("\nPercentage distribution by Department:")
print(pd.crosstab(df['Department'], df['Performance_Category'], normalize='index') * 100)


# ============================================================================
# 9. CORRELATION HEATMAP
# ============================================================================

print("\n" + "="*70)
print("9. CREATING CORRELATION HEATMAP")
print("="*70)

fig, ax = plt.subplots(figsize=(8, 6))

# Create correlation matrix
corr = df[numeric_cols].corr()

# Create heatmap
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

# Set ticks and labels
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
ax.set_yticklabels(numeric_cols)

# Add correlation values
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=10)

ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\correlation_heatmap.png', dpi=100, bbox_inches='tight')
print("\n✓ Correlation heatmap saved to 'correlation_heatmap.png'")
plt.close()


# ============================================================================
# 10. MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("10. MISSING VALUES ANALYSIS")
print("="*70)

print(f"\nMissing values per column:")
missing_vals = df.isnull().sum()
print(missing_vals)

if missing_vals.sum() == 0:
    print("✓ No missing values in the dataset!")
else:
    print(f"\nPercentage of missing values:")
    print((missing_vals / len(df) * 100).round(2))


# ============================================================================
# 11. COMPREHENSIVE EDA REPORT
# ============================================================================

print("\n" + "="*70)
print("11. COMPREHENSIVE EDA REPORT")
print("="*70)

def generate_eda_report(df):
    """
    Generate a comprehensive EDA report
    """
    report = f"""
    {'='*60}
    EXPLORATORY DATA ANALYSIS REPORT
    {'='*60}
    
    DATASET OVERVIEW:
    - Shape: {df.shape}
    - Memory usage: {df.memory_usage().sum() / 1024:.2f} KB
    
    DATA TYPES:
    {df.dtypes.to_string()}
    
    MISSING VALUES:
    {df.isnull().sum().to_string()}
    
    NUMERIC SUMMARY:
    {df.describe().to_string()}
    
    CATEGORICAL SUMMARY:
    {df.describe(include=['object']).to_string()}
    
    CORRELATION MATRIX:
    {df.corr().to_string()}
    
    KEY INSIGHTS:
    - Most correlated features with Salary: {df.corr()['Salary'].sort_values(ascending=False).index[1]}
    - Department with highest average salary: {df.groupby('Department')['Salary'].mean().idxmax()}
    - Average salary range: ${df['Salary'].min():,.0f} - ${df['Salary'].max():,.0f}
    - Most common department: {df['Department'].value_counts().idxmax()}
    """
    
    return report

report = generate_eda_report(df)
print(report)

# Save report
with open('e:\\Python_for_DS\\eda_report.txt', 'w') as f:
    f.write(report)
print("\n✓ EDA report saved to 'eda_report.txt'")


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Start with descriptive statistics
2. Visualize distributions and relationships
3. Check for outliers and anomalies
4. Analyze correlations between variables
5. Look for patterns and trends
6. Check data quality (missing values, duplicates)
7. Create hypotheses for further investigation
8. Document findings for stakeholders
9. Use EDA insights to guide feature engineering
10. Always visualize before modeling!
""")

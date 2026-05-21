"""
Data Loading and Exploration
=============================
The first step in any ML project: loading, understanding, and exploring your data.
This is crucial for building good models.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
import os

# ============================================================================
# 1. LOADING DATA FROM DIFFERENT SOURCES
# ============================================================================

print("\n" + "="*70)
print("1. LOADING DATA FROM DIFFERENT SOURCES")
print("="*70)

# Create sample CSV file for demonstration
sample_data = """Name,Age,Salary,Department,Years_Experience
Alice,25,50000,Sales,2
Bob,30,60000,IT,5
Charlie,35,75000,HR,8
David,40,80000,IT,10
Emma,28,55000,Sales,3
Frank,45,90000,IT,15
Grace,32,65000,HR,6"""

with open('e:\\Python_for_DS\\sample_data.csv', 'w') as f:
    f.write(sample_data)

# Load CSV file
df = pd.read_csv('e:\\Python_for_DS\\sample_data.csv')
print("\nLoaded CSV file:")
print(df)
print(f"Shape: {df.shape}")

# Read with specific parameters
df_subset = pd.read_csv('e:\\Python_for_DS\\sample_data.csv', nrows=3)
print("\nFirst 3 rows only:")
print(df_subset)

# Specify column types
df_types = pd.read_csv('e:\\Python_for_DS\\sample_data.csv',
                       dtype={'Age': int, 'Salary': float})
print(f"\nData types:\n{df_types.dtypes}")


# ============================================================================
# 2. DATA SHAPE AND BASIC INFO
# ============================================================================

print("\n" + "="*70)
print("2. DATA SHAPE AND BASIC INFO")
print("="*70)

print(f"\nNumber of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

print("\nFirst few rows:")
print(df.head())

print("\nLast few rows:")
print(df.tail(2))

print("\nDataFrame Info:")
df.info()

print("\nData types:")
print(df.dtypes)


# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "="*70)
print("3. DESCRIPTIVE STATISTICS")
print("="*70)

print("\nNumeric columns summary:")
print(df.describe())

print("\nDetailed statistics for Salary:")
print(f"Mean: {df['Salary'].mean():.2f}")
print(f"Median: {df['Salary'].median():.2f}")
print(f"Std Dev: {df['Salary'].std():.2f}")
print(f"Min: {df['Salary'].min()}")
print(f"Max: {df['Salary'].max()}")
print(f"Range: {df['Salary'].max() - df['Salary'].min()}")
print(f"Quartiles (25%, 50%, 75%): {df['Salary'].quantile([0.25, 0.5, 0.75]).tolist()}")


# ============================================================================
# 4. CHECKING FOR MISSING VALUES
# ============================================================================

print("\n" + "="*70)
print("4. CHECKING FOR MISSING VALUES")
print("="*70)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nPercentage of missing values:")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage)

print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Data with missing values
df_missing = df.copy()
df_missing.loc[0, 'Salary'] = np.nan
df_missing.loc[2, 'Age'] = np.nan

print("\nDataFrame with missing values:")
print(df_missing)
print(f"\nMissing values: \n{df_missing.isnull().sum()}")


# ============================================================================
# 5. DATA TYPES AND CONVERSIONS
# ============================================================================

print("\n" + "="*70)
print("5. DATA TYPES AND CONVERSIONS")
print("="*70)

print("\nCurrent data types:")
print(df.dtypes)

# Convert column types
df_converted = df.copy()
df_converted['Salary'] = df_converted['Salary'].astype('float64')
df_converted['Years_Experience'] = df_converted['Years_Experience'].astype('int64')

print("\nAfter type conversion:")
print(df_converted.dtypes)

# Convert to category (good for categorical variables)
df_converted['Department'] = df_converted['Department'].astype('category')
print("\nAfter converting Department to category:")
print(df_converted.dtypes)


# ============================================================================
# 6. UNIQUE VALUES AND VALUE COUNTS
# ============================================================================

print("\n" + "="*70)
print("6. UNIQUE VALUES AND VALUE COUNTS")
print("="*70)

print("\nUnique departments:")
print(df['Department'].unique())
print(f"Number of unique departments: {df['Department'].nunique()}")

print("\nDepartment distribution:")
print(df['Department'].value_counts())

print("\nDepartment distribution (%):")
print(df['Department'].value_counts(normalize=True) * 100)


# ============================================================================
# 7. IDENTIFYING DUPLICATES
# ============================================================================

print("\n" + "="*70)
print("7. IDENTIFYING DUPLICATES")
print("="*70)

# Create data with duplicates
df_duplicates = pd.concat([df.head(3), df.head(2)], ignore_index=True)
print("\nDataFrame with duplicates:")
print(df_duplicates)

print("\nCheck for duplicate rows:")
print(df_duplicates.duplicated())

print("\nNumber of duplicate rows:")
print(df_duplicates.duplicated().sum())

print("\nRemove duplicates:")
print(df_duplicates.drop_duplicates())


# ============================================================================
# 8. IDENTIFYING OUTLIERS
# ============================================================================

print("\n" + "="*70)
print("8. IDENTIFYING OUTLIERS")
print("="*70)

# Create data with outlier
df_outlier = df.copy()
df_outlier.loc[7] = ['Outlier Person', 100, 500000, 'IT', 50]

print("\nDataFrame with outlier (very high salary):")
print(df_outlier[['Name', 'Salary']])

# IQR method for outlier detection
Q1 = df_outlier['Salary'].quantile(0.25)
Q3 = df_outlier['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nQ1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Outlier bounds: [{lower_bound}, {upper_bound}]")

outliers = df_outlier[(df_outlier['Salary'] < lower_bound) | 
                      (df_outlier['Salary'] > upper_bound)]
print(f"\nDetected outliers:")
print(outliers)


# ============================================================================
# 9. CORRELATION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("9. CORRELATION ANALYSIS")
print("="*70)

print("\nNumeric columns only:")
numeric_cols = df.select_dtypes(include=[np.number])
print(numeric_cols.head())

print("\nCorrelation matrix:")
correlation = numeric_cols.corr()
print(correlation)

print("\nCorrelation with Salary:")
print(correlation['Salary'].sort_values(ascending=False))


# ============================================================================
# 10. DATA EXPLORATION SUMMARY
# ============================================================================

print("\n" + "="*70)
print("10. DATA EXPLORATION SUMMARY")
print("="*70)

def explore_data(dataframe):
    """
    Generate a comprehensive data exploration report
    """
    print(f"\n{'='*50}")
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print(f"{'='*50}")
    
    print(f"\nDataset Shape: {dataframe.shape}")
    print(f"\nColumn Names and Types:")
    for col in dataframe.columns:
        print(f"  {col}: {dataframe[col].dtype}")
    
    print(f"\nMissing Values:")
    missing = dataframe.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values!")
    else:
        print(missing[missing > 0])
    
    print(f"\nNumeric Summary:")
    print(dataframe.describe())
    
    print(f"\nCategorical Columns Summary:")
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        print(f"\n  {col}:")
        print(f"    Unique values: {dataframe[col].nunique()}")
        print(f"    Value counts:")
        print(f"    {dataframe[col].value_counts().to_string()}")

explore_data(df)


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Always start with df.head(), shape, and info()
2. Check for missing values early
3. Understand data types before modeling
4. Look for outliers and duplicates
5. Analyze distributions and correlations
6. Document interesting patterns
7. Create an exploration report for stakeholders
""")

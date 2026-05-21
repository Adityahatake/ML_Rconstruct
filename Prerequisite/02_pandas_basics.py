"""
Pandas Fundamentals for Data Science
=====================================
Pandas provides high-level data structures and tools for data manipulation.
DataFrames are like Excel sheets on steroids!

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np

# ============================================================================
# 1. CREATING DATAFRAMES
# ============================================================================

print("\n" + "="*70)
print("1. CREATING DATAFRAMES")
print("="*70)

# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Emma'],
    'Age': [25, 30, 35, 40, 28],
    'Salary': [50000, 60000, 75000, 80000, 55000],
    'Department': ['Sales', 'IT', 'HR', 'IT', 'Sales']
}

df = pd.DataFrame(data)
print("\nDataFrame from dictionary:")
print(df)

# From CSV (example)
# df = pd.read_csv('data.csv')

# From list of dictionaries
records = [
    {'product': 'Laptop', 'price': 1000, 'quantity': 5},
    {'product': 'Mouse', 'price': 25, 'quantity': 50},
    {'product': 'Monitor', 'price': 300, 'quantity': 10}
]
df_products = pd.DataFrame(records)
print("\nDataFrame from records:")
print(df_products)


# ============================================================================
# 2. BASIC DATAFRAME INFORMATION
# ============================================================================

print("\n" + "="*70)
print("2. DATAFRAME INFORMATION")
print("="*70)

print(f"\nShape: {df.shape}")  # (rows, columns)
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic info:")
df.info()
print(f"\nStatistical summary:")
print(df.describe())


# ============================================================================
# 3. SELECTING DATA
# ============================================================================

print("\n" + "="*70)
print("3. SELECTING DATA")
print("="*70)

print("\nSelect single column (Series):")
print(df['Name'])

print("\nSelect multiple columns:")
print(df[['Name', 'Age', 'Salary']])

print("\nSelect by position (iloc):")
print(df.iloc[0])  # First row
print("\nFirst 3 rows:")
print(df.iloc[0:3])

print("\nSelect by label (loc):")
print(df.loc[0, 'Name'])
print("\nRows with label 0-2:")
print(df.loc[0:2, ['Name', 'Salary']])


# ============================================================================
# 4. FILTERING DATA
# ============================================================================

print("\n" + "="*70)
print("4. FILTERING DATA")
print("="*70)

print("\nEmployees with Age > 30:")
print(df[df['Age'] > 30])

print("\nEmployees in Sales department:")
print(df[df['Department'] == 'Sales'])

print("\nMultiple conditions (Age > 25 AND Salary > 55000):")
print(df[(df['Age'] > 25) & (df['Salary'] > 55000)])

print("\nOR condition (Department is Sales OR IT):")
print(df[(df['Department'] == 'Sales') | (df['Department'] == 'IT')])


# ============================================================================
# 5. MODIFYING DATA
# ============================================================================

print("\n" + "="*70)
print("5. MODIFYING DATA")
print("="*70)

df_copy = df.copy()

# Add new column
df_copy['Bonus'] = df_copy['Salary'] * 0.1
print("\nAfter adding Bonus column:")
print(df_copy.head())

# Modify existing column
df_copy['Age'] = df_copy['Age'] + 1
print("\nAfter incrementing Age:")
print(df_copy[['Name', 'Age']].head())

# Rename columns
df_copy = df_copy.rename(columns={'Name': 'Employee_Name'})
print("\nAfter renaming 'Name' to 'Employee_Name':")
print(df_copy.columns.tolist())


# ============================================================================
# 6. SORTING AND GROUPING
# ============================================================================

print("\n" + "="*70)
print("6. SORTING AND GROUPING")
print("="*70)

print("\nSort by Salary (ascending):")
print(df.sort_values('Salary'))

print("\nSort by Age (descending):")
print(df.sort_values('Age', ascending=False))

print("\nGroupBy Department - Count employees:")
print(df.groupby('Department').size())

print("\nGroupBy Department - Average Salary:")
print(df.groupby('Department')['Salary'].mean())

print("\nGroupBy Department - Multiple aggregations:")
print(df.groupby('Department')['Salary'].agg(['mean', 'min', 'max', 'count']))


# ============================================================================
# 7. HANDLING MISSING DATA
# ============================================================================

print("\n" + "="*70)
print("7. HANDLING MISSING DATA")
print("="*70)

df_missing = pd.DataFrame({
    'Name': ['Alice', 'Bob', None, 'David', 'Emma'],
    'Age': [25, None, 35, 40, 28],
    'Salary': [50000, 60000, 75000, None, 55000]
})

print("\nDataFrame with missing values:")
print(df_missing)

print("\nCheck for missing values:")
print(df_missing.isnull())

print("\nCount missing values per column:")
print(df_missing.isnull().sum())

print("\nDrop rows with missing values:")
print(df_missing.dropna())

print("\nFill missing values:")
print(df_missing.fillna({'Age': df_missing['Age'].mean(), 
                        'Salary': df_missing['Salary'].mean()}))


# ============================================================================
# 8. MERGE AND CONCATENATE
# ============================================================================

print("\n" + "="*70)
print("8. MERGE AND CONCATENATE")
print("="*70)

df1 = pd.DataFrame({'ID': [1, 2, 3],
                    'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 3],
                    'Salary': [50000, 60000, 75000]})

print("\nDataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

print("\nMerge on ID:")
merged = pd.merge(df1, df2, on='ID')
print(merged)

df3 = pd.DataFrame({'ID': [4, 5],
                    'Name': ['David', 'Emma']})

print("\nConcatenate df1 and df3:")
concatenated = pd.concat([df1, df3], ignore_index=True)
print(concatenated)


# ============================================================================
# 9. APPLYING FUNCTIONS
# ============================================================================

print("\n" + "="*70)
print("9. APPLYING FUNCTIONS")
print("="*70)

df_apply = df.copy()

# Apply function to a column
df_apply['Name_Length'] = df_apply['Name'].apply(len)
print("\nAdd Name_Length column:")
print(df_apply[['Name', 'Name_Length']])

# Apply custom function
def salary_category(salary):
    if salary < 60000:
        return 'Low'
    elif salary < 75000:
        return 'Medium'
    else:
        return 'High'

df_apply['Salary_Category'] = df_apply['Salary'].apply(salary_category)
print("\nAdd Salary_Category column:")
print(df_apply[['Salary', 'Salary_Category']])


# ============================================================================
# 10. DATA STATISTICS AND CORRELATION
# ============================================================================

print("\n" + "="*70)
print("10. DATA STATISTICS AND CORRELATION")
print("="*70)

numeric_df = pd.DataFrame({
    'Math': [85, 90, 78, 92, 88],
    'Science': [88, 85, 80, 95, 92],
    'English': [80, 88, 85, 89, 85]
})

print("\nNumeric DataFrame:")
print(numeric_df)

print("\nCorrelation Matrix:")
print(numeric_df.corr())

print("\nValue counts (for categorical data):")
dept_counts = df['Department'].value_counts()
print(dept_counts)


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. DataFrames are 2D tables with labeled axes
2. Use loc[] for label-based indexing, iloc[] for position-based
3. Boolean indexing is powerful for filtering
4. GroupBy is essential for aggregation operations
5. Handle missing data appropriately for your use case
6. Merge and concatenate for combining datasets
7. Apply functions for custom transformations
""")

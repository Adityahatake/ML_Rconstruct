"""
Data Preprocessing and Cleaning
================================
Data preprocessing is crucial! Raw data is rarely clean and ready for modeling.
Spend 80% of your time here - it directly impacts model quality.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# ============================================================================
# CREATE SAMPLE DATASET WITH VARIOUS ISSUES
# ============================================================================

raw_data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [25, 30, np.nan, 40, 28, 35, 45, 22, 38, 29],
    'Salary': [50000, 60000, 75000, 80000, np.nan, 65000, 90000, 45000, 85000, 55000],
    'Department': ['Sales', 'IT', 'HR', 'IT', 'Sales', 'HR', 'IT', 'Sales', 'HR', 'IT'],
    'Experience': [2, 5, 8, 10, 3, 6, 15, 1, 9, 4],
    'Performance': ['Good', 'Excellent', 'Good', 'Excellent', 'Average', 'Good', 'Excellent', 'Good', 'Excellent', 'Average']
}

df = pd.DataFrame(raw_data)

print("\n" + "="*70)
print("ORIGINAL DATA WITH ISSUES")
print("="*70)
print("\nRaw Data:")
print(df)
print(f"\nMissing Values:\n{df.isnull().sum()}")


# ============================================================================
# 1. HANDLING MISSING VALUES
# ============================================================================

print("\n" + "="*70)
print("1. HANDLING MISSING VALUES")
print("="*70)

df_clean = df.copy()

# Strategy 1: Drop rows with missing values
print("\nStrategy 1: Drop rows with missing values")
print(f"Before: {len(df_clean)} rows")
df_drop = df_clean.dropna()
print(f"After: {len(df_drop)} rows")
print("❌ Not ideal - we lose data!")

# Strategy 2: Fill with mean (for numeric columns)
print("\nStrategy 2: Fill with mean/median")
df_clean = df.copy()
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].mean())
df_clean['Salary'] = df_clean['Salary'].fillna(df_clean['Salary'].median())
print(f"Age missing values: {df_clean['Age'].isnull().sum()}")
print(f"Salary missing values: {df_clean['Salary'].isnull().sum()}")

# Strategy 3: Forward/backward fill (for time series)
print("\nStrategy 3: Forward fill (for time series data)")
df_ffill = df.copy()
df_ffill['Age'] = df_ffill['Age'].fillna(method='ffill')
df_ffill['Salary'] = df_ffill['Salary'].fillna(method='ffill')


# ============================================================================
# 2. HANDLING DUPLICATES
# ============================================================================

print("\n" + "="*70)
print("2. HANDLING DUPLICATES")
print("="*70)

df_with_dup = pd.concat([df_clean, df_clean.iloc[0:2]], ignore_index=True)
print(f"\nRows with duplicates: {df_with_dup.duplicated().sum()}")

df_clean = df_clean.drop_duplicates(subset=['ID'])
print(f"After removing duplicates: {len(df_clean)} rows")


# ============================================================================
# 3. HANDLING OUTLIERS
# ============================================================================

print("\n" + "="*70)
print("3. HANDLING OUTLIERS")
print("="*70)

# Add outlier
df_clean.loc[5, 'Salary'] = 500000

print("\nOutlier Detection using IQR method:")
Q1 = df_clean['Salary'].quantile(0.25)
Q3 = df_clean['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
print(f"Bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")

outliers = df_clean[(df_clean['Salary'] < lower_bound) | 
                    (df_clean['Salary'] > upper_bound)]
print(f"\nOutliers detected:")
print(outliers)

# Handle outliers - replace with upper bound
df_clean.loc[df_clean['Salary'] > upper_bound, 'Salary'] = upper_bound
print(f"\nAfter capping outliers:")
print(df_clean[['ID', 'Salary']])


# ============================================================================
# 4. FEATURE SCALING
# ============================================================================

print("\n" + "="*70)
print("4. FEATURE SCALING")
print("="*70)

print("\nBefore scaling:")
print(f"Age range: {df_clean['Age'].min():.0f} - {df_clean['Age'].max():.0f}")
print(f"Salary range: {df_clean['Salary'].min():.0f} - {df_clean['Salary'].max():.0f}")

# Standardization (Z-score normalization)
scaler = StandardScaler()
df_standardized = df_clean.copy()
df_standardized[['Age', 'Salary']] = scaler.fit_transform(df_clean[['Age', 'Salary']])

print("\nAfter Standardization (Z-score):")
print(f"Age mean: {df_standardized['Age'].mean():.4f}, std: {df_standardized['Age'].std():.4f}")
print(f"Salary mean: {df_standardized['Salary'].mean():.4f}, std: {df_standardized['Salary'].std():.4f}")

# Normalization (Min-Max scaling)
scaler_minmax = MinMaxScaler()
df_normalized = df_clean.copy()
df_normalized[['Age', 'Salary']] = scaler_minmax.fit_transform(df_clean[['Age', 'Salary']])

print("\nAfter Normalization (Min-Max to [0,1]):")
print(f"Age range: {df_normalized['Age'].min():.4f} - {df_normalized['Age'].max():.4f}")
print(f"Salary range: {df_normalized['Salary'].min():.4f} - {df_normalized['Salary'].max():.4f}")


# ============================================================================
# 5. ENCODING CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*70)
print("5. ENCODING CATEGORICAL VARIABLES")
print("="*70)

df_encoded = df_clean.copy()

# Label Encoding (for ordinal categories)
print("\nLabel Encoding (Performance: Average=0, Good=1, Excellent=2):")
performance_map = {'Average': 0, 'Good': 1, 'Excellent': 2}
df_encoded['Performance_Encoded'] = df_encoded['Performance'].map(performance_map)
print(df_encoded[['Performance', 'Performance_Encoded']])

# One-Hot Encoding (for nominal categories)
print("\nOne-Hot Encoding (Department):")
department_dummies = pd.get_dummies(df_encoded['Department'], prefix='Dept')
print(department_dummies)

df_encoded = pd.concat([df_encoded, department_dummies], axis=1)
print("\nDataFrame with one-hot encoded Department:")
print(df_encoded.head())


# ============================================================================
# 6. LOG TRANSFORMATION
# ============================================================================

print("\n" + "="*70)
print("6. LOG TRANSFORMATION")
print("="*70)

print("\nOriginal Salary distribution (skewed):")
print(f"Skewness: {df_clean['Salary'].skew():.4f}")
print(f"Mean: {df_clean['Salary'].mean():.0f}, Median: {df_clean['Salary'].median():.0f}")

# Apply log transformation
df_log = df_clean.copy()
df_log['Salary_Log'] = np.log(df_log['Salary'])

print("\nAfter Log Transformation:")
print(f"Skewness: {df_log['Salary_Log'].skew():.4f}")
print(f"Mean: {df_log['Salary_Log'].mean():.4f}, Median: {df_log['Salary_Log'].median():.4f}")


# ============================================================================
# 7. CREATING DERIVED FEATURES
# ============================================================================

print("\n" + "="*70)
print("7. CREATING DERIVED FEATURES")
print("="*70)

df_features = df_clean.copy()

# Create new features
df_features['Salary_per_Year_Exp'] = df_features['Salary'] / (df_features['Experience'] + 1)
df_features['Age_Squared'] = df_features['Age'] ** 2
df_features['Is_Senior'] = (df_features['Experience'] >= 5).astype(int)
df_features['Is_High_Salary'] = (df_features['Salary'] > df_features['Salary'].median()).astype(int)

print("\nNew derived features:")
print(df_features[['ID', 'Salary_per_Year_Exp', 'Age_Squared', 'Is_Senior', 'Is_High_Salary']])


# ============================================================================
# 8. REMOVING UNNECESSARY FEATURES
# ============================================================================

print("\n" + "="*70)
print("8. REMOVING UNNECESSARY FEATURES")
print("="*70)

print("\nOriginal features:")
print(df_features.columns.tolist())

# Drop ID (not useful for prediction)
df_features = df_features.drop(['ID', 'Performance'], axis=1)

print("\nAfter removing unnecessary features:")
print(df_features.columns.tolist())


# ============================================================================
# 9. COMPLETE PREPROCESSING PIPELINE
# ============================================================================

print("\n" + "="*70)
print("9. COMPLETE PREPROCESSING PIPELINE")
print("="*70)

def preprocess_data(df_raw):
    """
    Complete data preprocessing pipeline
    """
    df = df_raw.copy()
    
    # Step 1: Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Salary'] = df['Salary'].fillna(df['Salary'].median())
    
    # Step 2: Remove duplicates
    df = df.drop_duplicates(subset=['ID'])
    
    # Step 3: Handle outliers (IQR method)
    Q1 = df['Salary'].quantile(0.25)
    Q3 = df['Salary'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df.loc[df['Salary'] > upper_bound, 'Salary'] = upper_bound
    
    # Step 4: Create derived features
    df['Salary_per_Exp'] = df['Salary'] / (df['Experience'] + 1)
    
    # Step 5: Encode categorical variables
    df['Performance_Encoded'] = df['Performance'].map({'Average': 0, 'Good': 1, 'Excellent': 2})
    df = pd.concat([df, pd.get_dummies(df['Department'], prefix='Dept')], axis=1)
    
    # Step 6: Drop unnecessary columns
    df = df.drop(['ID', 'Performance', 'Department'], axis=1)
    
    # Step 7: Feature scaling
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df

df_processed = preprocess_data(df)

print("\nPreprocessed data:")
print(df_processed.head())
print(f"\nShape: {df_processed.shape}")
print(f"\nData types:\n{df_processed.dtypes}")
print(f"\nMissing values: {df_processed.isnull().sum().sum()}")


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Handle missing values strategically (context matters!)
2. Remove duplicates before modeling
3. Detect and handle outliers appropriately
4. Scale features for algorithms that require it
5. Encode categorical variables correctly
6. Create derived features from domain knowledge
7. Remove unnecessary features
8. Create a reproducible preprocessing pipeline
9. Document all preprocessing steps
10. Save preprocessing parameters for production!
""")

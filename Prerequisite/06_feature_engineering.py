"""
Feature Engineering
===================
Feature engineering is an art + science. Good features beat fancy algorithms!
This is where domain knowledge matters most.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA

# ============================================================================
# CREATE SAMPLE DATASET
# ============================================================================

data = {
    'Age': np.random.randint(22, 60, 100),
    'Years_Experience': np.random.randint(0, 35, 100),
    'Salary': np.random.randint(40000, 150000, 100),
    'Education_Level': np.random.choice([1, 2, 3, 4], 100),  # 1=HS, 2=Bach, 3=Mast, 4=PhD
    'Department': np.random.choice(['Sales', 'IT', 'HR'], 100),
    'Performance_Score': np.random.randint(60, 100, 100),
    'Date_Joined': pd.date_range('2020-01-01', periods=100, freq='D')
}

df = pd.DataFrame(data)

print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)
print("\nOriginal dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())


# ============================================================================
# 1. POLYNOMIAL FEATURES
# ============================================================================

print("\n" + "="*70)
print("1. POLYNOMIAL FEATURES")
print("="*70)

print("\nOriginal features: Age, Years_Experience")
print("Sample data:")
print(df[['Age', 'Years_Experience']].head())

# Create polynomial features (degree 2: includes original + squares + interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
X = df[['Age', 'Years_Experience']].values
X_poly = poly.fit_transform(X)

print("\nAfter polynomial transformation (degree=2):")
print(f"Number of features: {X.shape[1]} → {X_poly.shape[1]}")
print(f"New features: Age, Experience, Age², Experience², Age×Experience")
print(f"Sample transformed data:\n{X_poly[:3]}")


# ============================================================================
# 2. INTERACTION FEATURES
# ============================================================================

print("\n" + "="*70)
print("2. INTERACTION FEATURES")
print("="*70)

df_interact = df.copy()

# Create interaction features
df_interact['Age_x_Experience'] = df_interact['Age'] * df_interact['Years_Experience']
df_interact['Age_x_Education'] = df_interact['Age'] * df_interact['Education_Level']
df_interact['Experience_x_Education'] = df_interact['Years_Experience'] * df_interact['Education_Level']

print("\nNew interaction features created:")
print(df_interact[['Age', 'Years_Experience', 'Age_x_Experience']].head())
print("\nCorrelation of interaction with Salary:")
print(f"Age × Experience: {df_interact['Age_x_Experience'].corr(df_interact['Salary']):.4f}")
print(f"Age × Education: {df_interact['Age_x_Education'].corr(df_interact['Salary']):.4f}")


# ============================================================================
# 3. BINNING AND DISCRETIZATION
# ============================================================================

print("\n" + "="*70)
print("3. BINNING AND DISCRETIZATION")
print("="*70)

df_binned = df.copy()

# Age binning
df_binned['Age_Group'] = pd.cut(df_binned['Age'], 
                                 bins=[0, 30, 40, 50, 100],
                                 labels=['Young', 'Middle', 'Senior', 'Veteran'])

# Salary binning
df_binned['Salary_Category'] = pd.qcut(df_binned['Salary'], 
                                        q=4, 
                                        labels=['Low', 'Medium', 'High', 'Very_High'],
                                        duplicates='drop')

print("\nAge binning:")
print(df_binned[['Age', 'Age_Group']].head(10))
print("\nAge group distribution:")
print(df_binned['Age_Group'].value_counts().sort_index())

print("\nSalary binning (quantile-based):")
print(df_binned[['Salary', 'Salary_Category']].head(10))


# ============================================================================
# 4. LOG TRANSFORMATION
# ============================================================================

print("\n" + "="*70)
print("4. LOG TRANSFORMATION")
print("="*70)

df_log = df.copy()

print(f"\nOriginal Salary - Skewness: {df_log['Salary'].skew():.4f}")
print(f"Original Salary - Mean: {df_log['Salary'].mean():.2f}, Median: {df_log['Salary'].median():.2f}")

# Apply log transformation
df_log['Salary_Log'] = np.log(df_log['Salary'])

print(f"\nLog-transformed Salary - Skewness: {df_log['Salary_Log'].skew():.4f}")
print(f"Log-transformed Salary - Mean: {df_log['Salary_Log'].mean():.4f}, Median: {df_log['Salary_Log'].median():.4f}")


# ============================================================================
# 5. ONE-HOT ENCODING
# ============================================================================

print("\n" + "="*70)
print("5. ONE-HOT ENCODING")
print("="*70)

print("\nOriginal Department column:")
print(df['Department'].unique())

# One-hot encoding
df_onehot = pd.get_dummies(df[['Department']], prefix='Dept')
print("\nAfter one-hot encoding:")
print(df_onehot.head())
print(f"Shape: {df_onehot.shape}")


# ============================================================================
# 6. ORDINAL ENCODING
# ============================================================================

print("\n" + "="*70)
print("6. ORDINAL ENCODING")
print("="*70)

df_ordinal = df.copy()

# Map education level to meaningful labels
education_map = {1: 'HighSchool', 2: 'Bachelor', 3: 'Master', 4: 'PhD'}
df_ordinal['Education_Label'] = df_ordinal['Education_Level'].map(education_map)

print("\nEducation level mapping:")
print(df_ordinal[['Education_Level', 'Education_Label']].drop_duplicates().sort_values('Education_Level'))


# ============================================================================
# 7. TEXT/DATE FEATURE EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("7. TEXT/DATE FEATURE EXTRACTION")
print("="*70)

df_date = df.copy()

# Extract date features
df_date['Year_Joined'] = df_date['Date_Joined'].dt.year
df_date['Month_Joined'] = df_date['Date_Joined'].dt.month
df_date['Quarter_Joined'] = df_date['Date_Joined'].dt.quarter
df_date['Day_of_Week'] = df_date['Date_Joined'].dt.day_name()
df_date['Days_Since_Join'] = (pd.Timestamp.now() - df_date['Date_Joined']).dt.days

print("\nDate feature extraction:")
print(df_date[['Date_Joined', 'Year_Joined', 'Month_Joined', 'Days_Since_Join']].head())


# ============================================================================
# 8. STANDARDIZATION AND SCALING
# ============================================================================

print("\n" + "="*70)
print("8. STANDARDIZATION AND SCALING")
print("="*70)

# Create a copy for scaling
df_scale = df[['Age', 'Years_Experience', 'Performance_Score', 'Salary']].copy()

print("\nBefore scaling:")
print(df_scale.describe())

# Standardize
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_scale)
df_scaled = pd.DataFrame(df_scaled, columns=df_scale.columns)

print("\nAfter standardization:")
print(df_scaled.describe())


# ============================================================================
# 9. CREATING AGGREGATE FEATURES
# ============================================================================

print("\n" + "="*70)
print("9. CREATING AGGREGATE FEATURES")
print("="*70)

# Group statistics
df_agg = df.copy()

# Department-level aggregates
dept_avg_salary = df_agg.groupby('Department')['Salary'].transform('mean')
dept_std_salary = df_agg.groupby('Department')['Salary'].transform('std')

df_agg['Dept_Avg_Salary'] = dept_avg_salary
df_agg['Dept_Std_Salary'] = dept_std_salary
df_agg['Salary_vs_Dept_Mean'] = df_agg['Salary'] - dept_avg_salary

print("\nAggregate features created:")
print(df_agg[['Department', 'Salary', 'Dept_Avg_Salary', 'Salary_vs_Dept_Mean']].head(10))


# ============================================================================
# 10. DIMENSIONALITY REDUCTION - PCA
# ============================================================================

print("\n" + "="*70)
print("10. DIMENSIONALITY REDUCTION - PCA")
print("="*70)

# Prepare data
X = df[['Age', 'Years_Experience', 'Performance_Score', 'Salary', 'Education_Level']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nOriginal features: {X_scaled.shape[1]}")

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print(f"After PCA: {X_pca.shape[1]} components")
print(f"\nExplained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
print(f"Total variance explained by 3 components: {sum(pca.explained_variance_ratio_):.2%}")


# ============================================================================
# 11. FEATURE SELECTION BASED ON CORRELATION
# ============================================================================

print("\n" + "="*70)
print("11. FEATURE SELECTION")
print("="*70)

numeric_df = df[['Age', 'Years_Experience', 'Salary', 'Education_Level', 'Performance_Score']].copy()

print("\nCorrelation with target (Salary):")
correlations = numeric_df.corr()['Salary'].sort_values(ascending=False)
print(correlations)

# Select features with correlation > 0.3
high_corr_features = correlations[abs(correlations) > 0.3].index.tolist()
print(f"\nFeatures with |correlation| > 0.3: {high_corr_features}")


# ============================================================================
# 12. COMPLETE FEATURE ENGINEERING PIPELINE
# ============================================================================

print("\n" + "="*70)
print("12. COMPLETE FEATURE ENGINEERING PIPELINE")
print("="*70)

def create_features(df):
    """
    Complete feature engineering pipeline
    """
    df_feat = df.copy()
    
    # 1. Create polynomial features
    df_feat['Age_Squared'] = df_feat['Age'] ** 2
    df_feat['Experience_Squared'] = df_feat['Years_Experience'] ** 2
    
    # 2. Create interaction features
    df_feat['Age_x_Experience'] = df_feat['Age'] * df_feat['Years_Experience']
    
    # 3. Create aggregate features
    dept_avg = df_feat.groupby('Department')['Salary'].transform('mean')
    df_feat['Salary_vs_Dept'] = df_feat['Salary'] - dept_avg
    
    # 4. Binning
    df_feat['Age_Group'] = pd.cut(df_feat['Age'], bins=[0, 30, 40, 50, 100],
                                   labels=['Young', 'Middle', 'Senior', 'Veteran'])
    
    # 5. One-hot encoding for categorical
    df_feat = pd.concat([df_feat, pd.get_dummies(df_feat['Department'], prefix='Dept')], axis=1)
    
    # 6. Drop original categorical columns
    df_feat = df_feat.drop(['Department', 'Date_Joined', 'Age_Group'], axis=1)
    
    # 7. Standardize numeric features
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_feat[numeric_cols] = scaler.fit_transform(df_feat[numeric_cols])
    
    return df_feat

df_final = create_features(df)

print("\nFinal engineered features:")
print(f"Original shape: {df.shape}")
print(f"Final shape: {df_final.shape}")
print(f"\nColumns: {df_final.columns.tolist()}")
print(f"\nFirst few rows:")
print(df_final.head())


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Polynomial features capture non-linear relationships
2. Interaction features combine domain knowledge
3. Binning can reveal non-linear patterns
4. Log transformation helps with skewed distributions
5. Encoding must match algorithm requirements
6. Aggregate features provide context
7. Dimensionality reduction fights curse of dimensionality
8. Feature selection improves model generalization
9. Always scale before using distance-based algorithms
10. Domain knowledge > fancy techniques!
11. Create features iteratively and validate
12. Document your feature engineering decisions
""")

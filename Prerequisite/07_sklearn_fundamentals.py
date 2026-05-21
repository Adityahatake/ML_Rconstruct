"""
Introduction to Scikit-Learn
============================
scikit-learn is the gold standard for ML in Python.
It provides consistent API for all algorithms.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ============================================================================
# CREATE SAMPLE DATASETS
# ============================================================================

print("\n" + "="*70)
print("SCIKIT-LEARN FUNDAMENTALS")
print("="*70)

# Regression dataset
print("\n1. Creating Regression Dataset")
np.random.seed(42)
X_reg = np.random.rand(100, 3) * 100
y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] - X_reg[:, 2] + np.random.randn(100) * 10

print(f"X shape: {X_reg.shape}, y shape: {y_reg.shape}")

# Classification dataset
print("\n2. Creating Classification Dataset")
from sklearn.datasets import make_classification
X_clf, y_clf = make_classification(n_samples=200, n_features=4, n_informative=2,
                                    n_redundant=0, n_classes=2, random_state=42)
print(f"X shape: {X_clf.shape}, y shape: {y_clf.shape}")
print(f"Class distribution: {np.bincount(y_clf)}")


# ============================================================================
# 1. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("1. TRAIN-TEST SPLIT")
print("="*70)

# For regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print(f"\nRegression dataset:")
print(f"  Training set: {X_train_reg.shape[0]} samples")
print(f"  Test set: {X_test_reg.shape[0]} samples")

# Stratified split for classification (maintains class distribution)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, stratify=y_clf, random_state=42
)

print(f"\nClassification dataset:")
print(f"  Training class distribution: {np.bincount(y_train_clf)}")
print(f"  Test class distribution: {np.bincount(y_test_clf)}")


# ============================================================================
# 2. FEATURE SCALING
# ============================================================================

print("\n" + "="*70)
print("2. FEATURE SCALING")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reg)
X_test_scaled = scaler.transform(X_test_reg)

print(f"\nBefore scaling - X_train mean: {X_train_reg.mean(axis=0)}")
print(f"Before scaling - X_train std: {X_train_reg.std(axis=0)}")

print(f"\nAfter scaling - X_train mean: {X_train_scaled.mean(axis=0)}")
print(f"After scaling - X_train std: {X_train_scaled.std(axis=0)}")

print("\n⚠️  IMPORTANT: Fit scaler ONLY on training data, then transform test data!")


# ============================================================================
# 3. LINEAR REGRESSION
# ============================================================================

print("\n" + "="*70)
print("3. LINEAR REGRESSION")
print("="*70)

# Train model
lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

print(f"\nModel coefficients: {lr.coef_}")
print(f"Model intercept: {lr.intercept_:.2f}")

# Make predictions
y_pred_train = lr.predict(X_train_reg)
y_pred_test = lr.predict(X_test_reg)

# Evaluate
mse_train = mean_squared_error(y_train_reg, y_pred_train)
mse_test = mean_squared_error(y_test_reg, y_pred_test)
r2_train = r2_score(y_train_reg, y_pred_train)
r2_test = r2_score(y_test_reg, y_pred_test)

print(f"\nTraining MSE: {mse_train:.4f}, R² Score: {r2_train:.4f}")
print(f"Test MSE: {mse_test:.4f}, R² Score: {r2_test:.4f}")


# ============================================================================
# 4. DECISION TREE REGRESSOR
# ============================================================================

print("\n" + "="*70)
print("4. DECISION TREE REGRESSOR")
print("="*70)

# Train with limited depth to prevent overfitting
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_reg.fit(X_train_reg, y_train_reg)

y_pred_train_dt = dt_reg.predict(X_train_reg)
y_pred_test_dt = dt_reg.predict(X_test_reg)

mse_train_dt = mean_squared_error(y_train_reg, y_pred_train_dt)
mse_test_dt = mean_squared_error(y_test_reg, y_pred_test_dt)
r2_train_dt = r2_score(y_train_reg, y_pred_train_dt)
r2_test_dt = r2_score(y_test_reg, y_pred_test_dt)

print(f"\nTraining MSE: {mse_train_dt:.4f}, R² Score: {r2_train_dt:.4f}")
print(f"Test MSE: {mse_test_dt:.4f}, R² Score: {r2_test_dt:.4f}")

print(f"\nFeature importances: {dt_reg.feature_importances_}")


# ============================================================================
# 5. RANDOM FOREST REGRESSOR
# ============================================================================

print("\n" + "="*70)
print("5. RANDOM FOREST REGRESSOR")
print("="*70)

rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

y_pred_train_rf = rf_reg.predict(X_train_reg)
y_pred_test_rf = rf_reg.predict(X_test_reg)

mse_train_rf = mean_squared_error(y_train_reg, y_pred_train_rf)
mse_test_rf = mean_squared_error(y_test_reg, y_pred_test_rf)
r2_train_rf = r2_score(y_train_reg, y_pred_train_rf)
r2_test_rf = r2_score(y_test_reg, y_pred_test_rf)

print(f"\nTraining MSE: {mse_train_rf:.4f}, R² Score: {r2_train_rf:.4f}")
print(f"Test MSE: {mse_test_rf:.4f}, R² Score: {r2_test_rf:.4f}")

print(f"\nFeature importances: {rf_reg.feature_importances_}")


# ============================================================================
# 6. LOGISTIC REGRESSION (CLASSIFICATION)
# ============================================================================

print("\n" + "="*70)
print("6. LOGISTIC REGRESSION (CLASSIFICATION)")
print("="*70)

# Scale features for logistic regression
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Train model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_clf_scaled, y_train_clf)

# Predictions
y_pred_train_lr = log_reg.predict(X_train_clf_scaled)
y_pred_test_lr = log_reg.predict(X_test_clf_scaled)

# Probabilities
y_proba_test_lr = log_reg.predict_proba(X_test_clf_scaled)

print(f"\nTraining Accuracy: {accuracy_score(y_train_clf, y_pred_train_lr):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred_test_lr):.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test_clf, y_pred_test_lr))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test_clf, y_pred_test_lr))


# ============================================================================
# 7. DECISION TREE CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("7. DECISION TREE CLASSIFIER")
print("="*70)

dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train_clf, y_train_clf)

y_pred_train_dt_clf = dt_clf.predict(X_train_clf)
y_pred_test_dt_clf = dt_clf.predict(X_test_clf)

print(f"\nTraining Accuracy: {accuracy_score(y_train_clf, y_pred_train_dt_clf):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred_test_dt_clf):.4f}")

print(f"\nFeature importances: {dt_clf.feature_importances_}")


# ============================================================================
# 8. RANDOM FOREST CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("8. RANDOM FOREST CLASSIFIER")
print("="*70)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)

y_pred_train_rf_clf = rf_clf.predict(X_train_clf)
y_pred_test_rf_clf = rf_clf.predict(X_test_clf)

print(f"\nTraining Accuracy: {accuracy_score(y_train_clf, y_pred_train_rf_clf):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test_clf, y_pred_test_rf_clf):.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test_clf, y_pred_test_rf_clf))


# ============================================================================
# 9. MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("9. MODEL COMPARISON - REGRESSION")
print("="*70)

models_comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
    'Train MSE': [mse_train, mse_train_dt, mse_train_rf],
    'Test MSE': [mse_test, mse_test_dt, mse_test_rf],
    'Train R²': [r2_train, r2_train_dt, r2_train_rf],
    'Test R²': [r2_test, r2_test_dt, r2_test_rf]
})

print("\n" + models_comparison.to_string(index=False))

best_model_idx = models_comparison['Test R²'].idxmax()
print(f"\n✓ Best model (by Test R²): {models_comparison.loc[best_model_idx, 'Model']}")


# ============================================================================
# 10. VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("10. VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Actual vs Predicted (Linear Regression)
axes[0, 0].scatter(y_test_reg, y_pred_test, alpha=0.6)
axes[0, 0].plot([y_test_reg.min(), y_test_reg.max()], 
                [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Linear Regression: Actual vs Predicted')

# Residuals (Linear Regression)
residuals = y_test_reg - y_pred_test
axes[0, 1].scatter(y_pred_test, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Linear Regression: Residual Plot')

# Model Comparison - MSE
axes[1, 0].bar(models_comparison['Model'], models_comparison['Test MSE'])
axes[1, 0].set_ylabel('Mean Squared Error')
axes[1, 0].set_title('Model Comparison: Test MSE')
axes[1, 0].tick_params(axis='x', rotation=45)

# Model Comparison - R²
axes[1, 1].bar(models_comparison['Model'], models_comparison['Test R²'])
axes[1, 1].set_ylabel('R² Score')
axes[1, 1].set_title('Model Comparison: Test R² Score')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\sklearn_models_comparison.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved to 'sklearn_models_comparison.png'")
plt.close()


# ============================================================================
# 11. COMMON SKLEARN PATTERNS
# ============================================================================

print("\n" + "="*70)
print("11. SCIKIT-LEARN WORKFLOW TEMPLATE")
print("="*70)

workflow = """
STANDARD SCIKIT-LEARN WORKFLOW:

1. IMPORT LIBRARIES
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score

2. LOAD AND PREPARE DATA
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

3. SCALE FEATURES (if needed)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)

4. INSTANTIATE MODEL
   model = RandomForestClassifier(n_estimators=100, random_state=42)

5. TRAIN MODEL
   model.fit(X_train, y_train)

6. MAKE PREDICTIONS
   y_pred = model.predict(X_test)

7. EVALUATE MODEL
   accuracy = accuracy_score(y_test, y_pred)

8. TUNE HYPERPARAMETERS (optional)
   - GridSearchCV or RandomizedSearchCV
   - Cross-validation

9. DEPLOY MODEL
   - Save model with joblib or pickle
   - Create prediction pipeline
"""

print(workflow)


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Train-test split prevents overfitting
2. Scale features BEFORE training (fit only on training data)
3. Different algorithms have different strengths
4. Always compare multiple models
5. Look at both training and test metrics (overfitting check)
6. Feature importance helps interpretability
7. Confusion matrix reveals classification errors
8. Cross-validation for robust evaluation
9. Hyperparameter tuning improves performance
10. Reproducibility: always set random_state!
11. sklearn has consistent API across algorithms
12. Read the documentation thoroughly!
""")

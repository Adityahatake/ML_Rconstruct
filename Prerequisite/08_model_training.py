"""
Model Training and Hyperparameter Tuning
========================================
Training good models requires careful tuning and validation.
This is both art and science!

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# ============================================================================
# CREATE SAMPLE DATASET
# ============================================================================

print("\n" + "="*70)
print("MODEL TRAINING AND HYPERPARAMETER TUNING")
print("="*70)

X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_redundant=2, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                      random_state=42, stratify=y)

print(f"\nDataset prepared:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Number of features: {X_train.shape[1]}")


# ============================================================================
# 1. BASELINE MODEL
# ============================================================================

print("\n" + "="*70)
print("1. BASELINE MODEL")
print("="*70)

baseline_model = LogisticRegression(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_pred = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"\nLogistic Regression (baseline):")
print(f"  Accuracy: {baseline_accuracy:.4f}")


# ============================================================================
# 2. CROSS-VALIDATION
# ============================================================================

print("\n" + "="*70)
print("2. CROSS-VALIDATION")
print("="*70)

print("\nCross-validation helps estimate true model performance:")

# 5-fold cross-validation
model_cv = DecisionTreeClassifier(max_depth=5, random_state=42)
cv_scores = cross_val_score(model_cv, X_train, y_train, cv=5, scoring='accuracy')

print(f"\n5-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 10-fold cross-validation
cv_scores_10 = cross_val_score(model_cv, X_train, y_train, cv=10, scoring='accuracy')
print(f"\n10-Fold CV Mean Score: {cv_scores_10.mean():.4f} (+/- {cv_scores_10.std():.4f})")


# ============================================================================
# 3. GRID SEARCH
# ============================================================================

print("\n" + "="*70)
print("3. GRID SEARCH")
print("="*70)

# Define hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("\nSearching hyperparameters:")
print(f"  max_depth: {param_grid['max_depth']}")
print(f"  min_samples_split: {param_grid['min_samples_split']}")
print(f"  min_samples_leaf: {param_grid['min_samples_leaf']}")
print(f"  Total combinations: {np.prod([len(v) for v in param_grid.values()])}")

# Grid Search
dt_model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Predictions with best model
best_model_gs = grid_search.best_estimator_
y_pred_gs = best_model_gs.predict(X_test)
accuracy_gs = accuracy_score(y_test, y_pred_gs)
print(f"Test Set Accuracy: {accuracy_gs:.4f}")

# Display top 10 results
results_df = pd.DataFrame(grid_search.cv_results_)
results_top10 = results_df[['param_max_depth', 'param_min_samples_split', 
                            'param_min_samples_leaf', 'mean_test_score', 'std_test_score']].head(10)
print(f"\nTop 10 hyperparameter combinations:")
print(results_top10.to_string(index=False))


# ============================================================================
# 4. RANDOMIZED SEARCH
# ============================================================================

print("\n" + "="*70)
print("4. RANDOMIZED SEARCH")
print("="*70)

# Randomized search is faster for large parameter spaces
from scipy.stats import randint

param_dist = {
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

print("\nRandomized search on parameter distributions:")

random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_dist,
    n_iter=20,  # Sample only 20 combinations
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"\nBest parameters (Randomized): {random_search.best_params_}")
print(f"Best CV Score: {random_search.best_score_:.4f}")

y_pred_rs = random_search.best_estimator_.predict(X_test)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
print(f"Test Set Accuracy: {accuracy_rs:.4f}")


# ============================================================================
# 5. RANDOM FOREST TUNING
# ============================================================================

print("\n" + "="*70)
print("5. RANDOM FOREST TUNING")
print("="*70)

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

print(f"\nBest parameters (Random Forest): {rf_grid_search.best_params_}")
print(f"Best CV Score: {rf_grid_search.best_score_:.4f}")

y_pred_rf = rf_grid_search.best_estimator_.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Test Set Accuracy: {accuracy_rf:.4f}")

print(f"\nFeature importances:")
feature_importance = rf_grid_search.best_estimator_.feature_importances_
for i, imp in enumerate(feature_importance):
    print(f"  Feature {i}: {imp:.4f}")


# ============================================================================
# 6. LEARNING CURVES
# ============================================================================

print("\n" + "="*70)
print("6. LEARNING CURVES")
print("="*70)

from sklearn.model_selection import learning_curve

model_lc = DecisionTreeClassifier(max_depth=7, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    model_lc, X_train, y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy',
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print("\nLearning curve analysis:")
print(f"Training sizes: {train_sizes}")
print(f"\nWith more training data:")
print(f"  Training accuracy plateaus at: {train_mean[-1]:.4f}")
print(f"  Validation accuracy reaches: {val_mean[-1]:.4f}")
print(f"  Gap between train and val: {train_mean[-1] - val_mean[-1]:.4f}")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score', linewidth=2)
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score', linewidth=2)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend()
plt.grid(True)
plt.savefig('e:\\Python_for_DS\\learning_curve.png', dpi=100, bbox_inches='tight')
print("\n✓ Learning curve saved to 'learning_curve.png'")
plt.close()


# ============================================================================
# 7. VALIDATION CURVE
# ============================================================================

print("\n" + "="*70)
print("7. VALIDATION CURVE")
print("="*70)

from sklearn.model_selection import validation_curve

param_name = 'max_depth'
param_range = range(1, 15)

train_scores_vc, val_scores_vc = validation_curve(
    DecisionTreeClassifier(random_state=42),
    X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean_vc = np.mean(train_scores_vc, axis=1)
train_std_vc = np.std(train_scores_vc, axis=1)
val_mean_vc = np.mean(val_scores_vc, axis=1)
val_std_vc = np.std(val_scores_vc, axis=1)

# Plot validation curve
plt.figure(figsize=(10, 6))
plt.plot(param_range, train_mean_vc, 'o-', color='blue', label='Training score', linewidth=2)
plt.fill_between(param_range, train_mean_vc - train_std_vc, train_mean_vc + train_std_vc, alpha=0.1, color='blue')
plt.plot(param_range, val_mean_vc, 'o-', color='red', label='Validation score', linewidth=2)
plt.fill_between(param_range, val_mean_vc - val_std_vc, val_mean_vc + val_std_vc, alpha=0.1, color='red')
plt.xlabel(f'max_depth')
plt.ylabel('Accuracy Score')
plt.title('Validation Curve')
plt.legend()
plt.grid(True)
plt.savefig('e:\\Python_for_DS\\validation_curve.png', dpi=100, bbox_inches='tight')
print(f"\n✓ Validation curve saved to 'validation_curve.png'")
plt.close()

print(f"\nOptimal max_depth: {param_range[np.argmax(val_mean_vc)]}")


# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("8. FINAL MODEL COMPARISON")
print("="*70)

# Train different models with tuned hyperparameters
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': grid_search.best_estimator_,
    'Random Forest': rf_grid_search.best_estimator_,
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Evaluate
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    results.append({
        'Model': name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Overfitting Gap': train_acc - test_acc
    })

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

best_model_name = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
print(f"\n✓ Best model: {best_model_name}")


# ============================================================================
# 9. HYPERPARAMETER TUNING BEST PRACTICES
# ============================================================================

print("\n" + "="*70)
print("9. HYPERPARAMETER TUNING BEST PRACTICES")
print("="*70)

best_practices = """
HYPERPARAMETER TUNING STRATEGIES:

1. START SIMPLE
   - Begin with default hyperparameters
   - Get a baseline score
   - Then improve incrementally

2. UNDERSTAND YOUR PARAMETERS
   - Read the documentation
   - Understand what each parameter controls
   - Common parameters: learning_rate, max_depth, n_estimators

3. USE CROSS-VALIDATION
   - Prevents overfitting to validation set
   - Gives more robust estimates
   - 5-10 fold is typical

4. GRID SEARCH vs RANDOM SEARCH
   - Grid Search: exhaustive, good for small spaces
   - Random Search: efficient for large spaces
   - Start with Random Search, narrow down with Grid Search

5. VALIDATION CURVES
   - Visualize parameter effects
   - Identify optimal parameter range
   - Detect overfitting/underfitting

6. LEARNING CURVES
   - Diagnose model performance issues
   - Determine if you need more data
   - Check for high variance or bias

7. COMPUTATIONAL BUDGET
   - Balance tuning time vs performance gain
   - Use n_jobs=-1 for parallelization
   - Consider early stopping for iterative algorithms

8. MAINTAIN A RECORD
   - Log all experiments
   - Track hyperparameters and scores
   - Document insights learned

9. AVOID DATA LEAKAGE
   - Scale ONLY on training data
   - Fit preprocessors ONLY on training data
   - Use cross-validation properly

10. FINAL VALIDATION
    - Use a separate test set (never touch during tuning!)
    - Report performance on test set
    - Check generalization
"""

print(best_practices)


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Cross-validation provides robust performance estimates
2. Grid Search exhaustive, Randomized Search efficient
3. Learning curves diagnose bias/variance problems
4. Always validate on test set at the end
5. Overfitting gap (train - test) indicates overfitting
6. Hyperparameter tuning is iterative
7. Monitor both train and test performance
8. Document your experiments!
9. Early stopping saves computation time
10. Parameter interactions matter - tune together!
""")

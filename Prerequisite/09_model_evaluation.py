"""
Model Evaluation and Metrics
=============================
Choosing the right metrics is crucial for assessing model performance.
Different problems require different metrics.

Author: ML Teacher (5 years experience)
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, auc, mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import seaborn as sns

# ============================================================================
# CREATE SAMPLE DATASETS
# ============================================================================

print("\n" + "="*70)
print("MODEL EVALUATION AND METRICS")
print("="*70)

# Classification dataset
X_clf, y_clf = make_classification(n_samples=300, n_features=5, n_informative=3,
                                    n_classes=2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Regression dataset
X_reg, y_reg = make_regression(n_samples=300, n_features=5, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train models
clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)
y_proba_clf = clf_model.predict_proba(X_test_clf)[:, 1]

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)


# ============================================================================
# 1. CLASSIFICATION METRICS
# ============================================================================

print("\n" + "="*70)
print("1. CLASSIFICATION METRICS")
print("="*70)

accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"  - Correct predictions / Total predictions")
print(f"  - Best for: Balanced datasets")

print(f"\nPrecision: {precision:.4f}")
print(f"  - TP / (TP + FP)")
print(f"  - Best for: Minimizing false positives")
print(f"  - Use when: False positives are costly")

print(f"\nRecall: {recall:.4f}")
print(f"  - TP / (TP + FN)")
print(f"  - Best for: Minimizing false negatives")
print(f"  - Use when: False negatives are costly")

print(f"\nF1 Score: {f1:.4f}")
print(f"  - Harmonic mean of Precision and Recall")
print(f"  - Best for: Imbalanced datasets")
print(f"  - Range: 0 to 1 (1 is perfect)")


# ============================================================================
# 2. CONFUSION MATRIX
# ============================================================================

print("\n" + "="*70)
print("2. CONFUSION MATRIX")
print("="*70)

cm = confusion_matrix(y_test_clf, y_pred_clf)
print(f"\nConfusion Matrix:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

# Calculate additional metrics from confusion matrix
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nSpecificity (True Negative Rate): {specificity:.4f}")
print(f"Sensitivity (Recall/True Positive Rate): {sensitivity:.4f}")


# ============================================================================
# 3. CLASSIFICATION REPORT
# ============================================================================

print("\n" + "="*70)
print("3. CLASSIFICATION REPORT")
print("="*70)

print("\nDetailed Classification Report:")
print(classification_report(y_test_clf, y_pred_clf))


# ============================================================================
# 4. ROC CURVE AND AUC
# ============================================================================

print("\n" + "="*70)
print("4. ROC CURVE AND AUC")
print("="*70)

roc_auc = roc_auc_score(y_test_clf, y_proba_clf)
fpr, tpr, thresholds = roc_curve(y_test_clf, y_proba_clf)

print(f"\nROC AUC Score: {roc_auc:.4f}")
print(f"  - AUC > 0.5: Better than random")
print(f"  - AUC > 0.7: Good")
print(f"  - AUC > 0.8: Very Good")
print(f"  - AUC > 0.9: Excellent")

# Plot ROC curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend(loc="lower right")
axes[0].grid(True, alpha=0.3)

# Confusion Matrix Heatmap
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\classification_metrics.png', dpi=100, bbox_inches='tight')
print("\n✓ Classification metrics visualization saved")
plt.close()


# ============================================================================
# 5. PRECISION-RECALL CURVE
# ============================================================================

print("\n" + "="*70)
print("5. PRECISION-RECALL CURVE")
print("="*70)

precision_curve, recall_curve, _ = precision_recall_curve(y_test_clf, y_proba_clf)
pr_auc = auc(recall_curve, precision_curve)

print(f"\nPR AUC Score: {pr_auc:.4f}")
print("  - More informative than ROC for imbalanced datasets")

# Plot PR curve
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\precision_recall_curve.png', dpi=100, bbox_inches='tight')
print("\n✓ Precision-Recall curve saved")
plt.close()


# ============================================================================
# 6. REGRESSION METRICS
# ============================================================================

print("\n" + "="*70)
print("6. REGRESSION METRICS")
print("="*70)

mse = mean_squared_error(y_test_reg, y_pred_reg)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_reg)
explained_var = explained_variance_score(y_test_reg, y_pred_reg)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"  - Average squared difference")
print(f"  - Penalizes large errors more")
print(f"  - Lower is better")

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.4f}")
print(f"  - Same units as target variable")
print(f"  - Popular for interpretation")

print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"  - Average absolute difference")
print(f"  - More robust to outliers than MSE")
print(f"  - Same units as target")

print(f"\nR² Score: {r2:.4f}")
print(f"  - Proportion of variance explained")
print(f"  - Range: -∞ to 1 (1 is perfect)")
print(f"  - >0.7: Good, >0.9: Excellent")

print(f"\nExplained Variance Score: {explained_var:.4f}")
print(f"  - Variance explained by model")


# ============================================================================
# 7. RESIDUAL ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("7. RESIDUAL ANALYSIS")
print("="*70)

residuals = y_test_reg - y_pred_reg

print(f"\nResidual Statistics:")
print(f"  Mean: {np.mean(residuals):.4f} (should be close to 0)")
print(f"  Std Dev: {np.std(residuals):.4f}")
print(f"  Min: {np.min(residuals):.4f}")
print(f"  Max: {np.max(residuals):.4f}")

# Plot residuals
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residual plot
axes[0].scatter(y_pred_reg, residuals, alpha=0.6)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Predicted Values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residual Plot')
axes[0].grid(True, alpha=0.3)

# Histogram of residuals
axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Residuals')
axes[1].axvline(x=0, color='r', linestyle='--')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\residual_analysis.png', dpi=100, bbox_inches='tight')
print("\n✓ Residual analysis plots saved")
plt.close()


# ============================================================================
# 8. ACTUAL VS PREDICTED
# ============================================================================

print("\n" + "="*70)
print("8. ACTUAL VS PREDICTED")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Regression
axes[0].scatter(y_test_reg, y_pred_reg, alpha=0.6)
axes[0].plot([y_test_reg.min(), y_test_reg.max()], 
             [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Values')
axes[0].set_ylabel('Predicted Values')
axes[0].set_title(f'Regression: Actual vs Predicted (R² = {r2:.3f})')
axes[0].grid(True, alpha=0.3)

# Classification: Show class distribution
class_distribution = pd.DataFrame({
    'True Label': ['Negative', 'Positive'],
    'Predicted Negative': [tn, fn],
    'Predicted Positive': [fp, tp]
})

class_distribution.set_index('True Label').plot(kind='bar', ax=axes[1])
axes[1].set_ylabel('Count')
axes[1].set_title('Classification Predictions')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
axes[1].legend(title='Prediction')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('e:\\Python_for_DS\\actual_vs_predicted.png', dpi=100, bbox_inches='tight')
print("\n✓ Actual vs Predicted plots saved")
plt.close()


# ============================================================================
# 9. METRIC SELECTION GUIDE
# ============================================================================

print("\n" + "="*70)
print("9. METRIC SELECTION GUIDE")
print("="*70)

guide = """
CLASSIFICATION METRICS SELECTION:

1. BALANCED DATASET → Accuracy
   - Roughly equal classes
   - Simple metric to understand

2. IMBALANCED DATASET → F1 Score or ROC AUC
   - Rare positive class
   - F1 balances precision and recall

3. FALSE POSITIVES COSTLY → Precision
   - Spam detection (don't want spam in inbox)
   - Fraud detection (don't want rejecting good customers)

4. FALSE NEGATIVES COSTLY → Recall
   - Disease diagnosis (don't want missing sick people)
   - Security threat detection (don't miss threats)

5. NEED THRESHOLD OPTIMIZATION → ROC AUC or PR AUC
   - Adjust prediction threshold
   - PR AUC better for imbalanced data

REGRESSION METRICS SELECTION:

1. GENERAL PURPOSE → RMSE or MAE
   - RMSE: Default choice, penalizes large errors
   - MAE: More robust to outliers

2. PERCENTAGE ERROR → R² Score
   - Variance explained
   - Easy to interpret (% of variance)

3. OUTLIERS PRESENT → MAE or Explained Variance
   - MSE/RMSE penalize outliers too much
   - MAE is more robust

4. NEED UNIT CONSISTENCY → RMSE or MAE
   - Same units as target variable
   - R² is unitless
"""

print(guide)


# ============================================================================
# 10. COMPREHENSIVE EVALUATION REPORT
# ============================================================================

print("\n" + "="*70)
print("10. COMPREHENSIVE EVALUATION REPORT")
print("="*70)

def generate_evaluation_report(y_true, y_pred, y_proba=None, task='classification'):
    """Generate comprehensive evaluation report"""
    
    if task == 'classification':
        report = f"""
        CLASSIFICATION MODEL EVALUATION REPORT
        {'='*50}
        
        PERFORMANCE METRICS:
        - Accuracy:  {accuracy_score(y_true, y_pred):.4f}
        - Precision: {precision_score(y_true, y_pred):.4f}
        - Recall:    {recall_score(y_true, y_pred):.4f}
        - F1 Score:  {f1_score(y_true, y_pred):.4f}
        
        DETAILED CLASSIFICATION REPORT:
        {classification_report(y_true, y_pred)}
        
        CONFUSION MATRIX:
        {confusion_matrix(y_true, y_pred)}
        """
        
        if y_proba is not None:
            roc_auc = roc_auc_score(y_true, y_proba)
            report += f"\n        ROC AUC Score: {roc_auc:.4f}"
    
    return report

report = generate_evaluation_report(y_test_clf, y_pred_clf, y_proba_clf)
print(report)


print("\n" + "="*70)
print("Key Takeaways:")
print("="*70)
print("""
1. Choose metrics based on problem type and business goals
2. Confusion matrix shows where model makes mistakes
3. Precision-Recall for imbalanced data
4. ROC AUC for threshold tuning
5. R² doesn't tell whole story - check residuals
6. Residuals should be random and centered at 0
7. Multiple metrics give better picture
8. Visualizations reveal patterns in errors
9. Consider business cost of different error types
10. Report both train and test metrics
11. Document which metrics were used and why
12. Track metrics over time to monitor drift
""")

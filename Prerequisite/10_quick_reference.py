"""
10. Complete ML Learning Path - Quick Reference
===============================================
Your comprehensive guide to machine learning fundamentals.
Act as reference material while working through the other files.

Author: ML Teacher (5 years experience)
Date: 2026
"""

# ============================================================================
# COMPLETE MACHINE LEARNING WORKFLOW
# ============================================================================

COMPLETE_ML_WORKFLOW = """
╔════════════════════════════════════════════════════════════════════════╗
║                   MACHINE LEARNING WORKFLOW                           ║
╚════════════════════════════════════════════════════════════════════════╝

1. PROBLEM DEFINITION & DATA COLLECTION
   └─ Understand business problem
   └─ Identify target variable
   └─ Collect relevant data
   └─ Assess data quality

2. EXPLORATORY DATA ANALYSIS (EDA) 
   └─ Load data and inspect (shape, types)
   └─ Check for missing values
   └─ Analyze distributions
   └─ Identify outliers
   └─ Explore correlations
   └─ Create visualizations
   └─ Document findings

3. DATA PREPROCESSING & CLEANING
   └─ Handle missing values (drop/fill/interpolate)
   └─ Remove duplicates
   └─ Handle outliers
   └─ Treat skewed distributions
   └─ Encode categorical variables
   └─ Handle imbalanced classes (if classification)

4. FEATURE ENGINEERING
   └─ Create polynomial features
   └─ Create interaction features
   └─ Create aggregate features
   └─ Binning/discretization
   └─ Log transformation
   └─ Feature scaling (StandardScaler)
   └─ Feature selection

5. TRAIN-TEST SPLIT
   └─ Split data (typically 80-20)
   └─ Use stratified split for imbalanced data
   └─ Important: NEVER touch test data during training!

6. MODEL SELECTION & TRAINING
   └─ Choose appropriate algorithms
   └─ Train models on training data
   └─ Use cross-validation for robust estimates
   └─ Hyperparameter tuning (Grid/Random Search)

7. MODEL EVALUATION
   └─ Choose appropriate metrics
   └─ Evaluate on test set
   └─ Compare multiple models
   └─ Check for overfitting/underfitting
   └─ Create visualizations (ROC, Learning curves)

8. MODEL DEPLOYMENT & MONITORING
   └─ Save trained model
   └─ Create prediction pipeline
   └─ Monitor performance in production
   └─ Retrain as needed

"""


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

KEY_CONCEPTS = """
╔════════════════════════════════════════════════════════════════════════╗
║                      KEY ML CONCEPTS                                  ║
╚════════════════════════════════════════════════════════════════════════╝

SUPERVISED LEARNING (labeled data)
├─ REGRESSION (continuous output)
│  ├─ Linear Regression
│  ├─ Decision Tree Regressor
│  ├─ Random Forest Regressor
│  ├─ Gradient Boosting Regressor
│  └─ Neural Networks
│
└─ CLASSIFICATION (discrete output)
   ├─ Logistic Regression
   ├─ Decision Tree Classifier
   ├─ Random Forest Classifier
   ├─ Gradient Boosting Classifier
   ├─ Support Vector Machines (SVM)
   ├─ Naive Bayes
   └─ Neural Networks

UNSUPERVISED LEARNING (unlabeled data)
├─ CLUSTERING (group similar items)
│  ├─ K-Means
│  ├─ Hierarchical Clustering
│  ├─ DBSCAN
│  └─ Gaussian Mixture Models
│
└─ DIMENSIONALITY REDUCTION (reduce features)
   ├─ Principal Component Analysis (PCA)
   ├─ t-SNE
   ├─ Feature Selection
   └─ Autoencoder

SEMI-SUPERVISED LEARNING (partially labeled data)
└─ Combines supervised and unsupervised approaches


BIAS-VARIANCE TRADEOFF
├─ HIGH BIAS (Underfitting)
│  └─ Model too simple
│  └─ Poor training performance
│  └─ Solution: More complex model, more features
│
├─ HIGH VARIANCE (Overfitting)
│  └─ Model too complex
│  └─ Great training, poor test performance
│  └─ Solution: Simpler model, more data, regularization
│
└─ OPTIMAL (Good Fit)
   └─ Low bias, low variance
   └─ Good training AND test performance


REGULARIZATION (prevent overfitting)
├─ L1 Regularization (Lasso)
│  └─ Reduces some coefficients to exactly zero
│  └─ Feature selection built-in
│
├─ L2 Regularization (Ridge)
│  └─ Shrinks all coefficients proportionally
│  └─ Keeps all features
│
└─ Dropout / Early Stopping / Cross-validation

"""


# ============================================================================
# COMMON ALGORITHMS CHEATSHEET
# ============================================================================

ALGORITHMS_GUIDE = """
╔════════════════════════════════════════════════════════════════════════╗
║                    ALGORITHMS SELECTION GUIDE                         ║
╚════════════════════════════════════════════════════════════════════════╝

REGRESSION:
┌─ Linear Regression
│  └─ Use when: Relationship is linear
│  └─ Pros: Fast, interpretable
│  └─ Cons: Assumes linearity
│  └─ Hyperparameters: fit_intercept, normalize
│
├─ Decision Tree
│  └─ Use when: Non-linear relationships
│  └─ Pros: Interpretable, handles non-linearity
│  └─ Cons: Can overfit
│  └─ Hyperparameters: max_depth, min_samples_split
│
├─ Random Forest
│  └─ Use when: Need robust predictions
│  └─ Pros: Reduces overfitting, feature importance
│  └─ Cons: Less interpretable than trees
│  └─ Hyperparameters: n_estimators, max_depth
│
└─ Gradient Boosting
   └─ Use when: Need best possible performance
   └─ Pros: Often wins Kaggle competitions
   └─ Cons: Slower, requires careful tuning
   └─ Hyperparameters: learning_rate, n_estimators

CLASSIFICATION:
┌─ Logistic Regression
│  └─ Use when: Linear decision boundary
│  └─ Pros: Fast, probability output, interpretable
│  └─ Cons: Assumes linear relationship
│  └─ Hyperparameters: C (regularization strength)
│
├─ Decision Tree
│  └─ Use when: Complex decision boundaries
│  └─ Pros: Interpretable, handles non-linearity
│  └─ Cons: Prone to overfitting
│  └─ Hyperparameters: max_depth, min_samples_split
│
├─ Random Forest
│  └─ Use when: Need robust, general classifier
│  └─ Pros: Handles multiple features well
│  └─ Cons: Harder to interpret
│  └─ Hyperparameters: n_estimators, max_depth
│
├─ Gradient Boosting
│  └─ Use when: Best accuracy is priority
│  └─ Pros: State-of-the-art performance
│  └─ Cons: Tuning-heavy, slow
│  └─ Hyperparameters: learning_rate, n_estimators
│
├─ Support Vector Machines (SVM)
│  └─ Use when: High-dimensional data
│  └─ Pros: Works well with many features
│  └─ Cons: Slow on large datasets, hard to tune
│  └─ Hyperparameters: C, kernel, gamma
│
└─ Naive Bayes
   └─ Use when: Text classification, baseline
   └─ Pros: Fast, works well with text
   └─ Cons: Assumes feature independence
   └─ Hyperparameters: alpha (smoothing)

"""


# ============================================================================
# IMPORTANT METRICS SUMMARY
# ============================================================================

METRICS_GUIDE = """
╔════════════════════════════════════════════════════════════════════════╗
║                         METRICS SUMMARY                               ║
╚════════════════════════════════════════════════════════════════════════╝

CLASSIFICATION METRICS:

Accuracy
├─ (TP + TN) / (TP + TN + FP + FN)
├─ Use when: Balanced classes
└─ Range: 0 to 1 (1 is perfect)

Precision (Positive Predictive Value)
├─ TP / (TP + FP)
├─ "Of positive predictions, how many were correct?"
├─ Use when: False positives are costly
└─ Range: 0 to 1 (1 is perfect)

Recall (Sensitivity, True Positive Rate)
├─ TP / (TP + FN)
├─ "Of actual positives, how many did we find?"
├─ Use when: False negatives are costly
└─ Range: 0 to 1 (1 is perfect)

F1 Score
├─ 2 * (Precision * Recall) / (Precision + Recall)
├─ Harmonic mean of precision and recall
├─ Use when: Imbalanced classes, need both metrics
└─ Range: 0 to 1 (1 is perfect)

ROC AUC
├─ Area under Receiver Operating Characteristic curve
├─ Measures performance across all thresholds
├─ Use when: Need to tune classification threshold
└─ Range: 0 to 1 (0.5 = random, 1 = perfect)

Confusion Matrix
├─ True Positives (TP): Correct positive predictions
├─ True Negatives (TN): Correct negative predictions
├─ False Positives (FP): Wrong positive predictions
└─ False Negatives (FN): Wrong negative predictions

REGRESSION METRICS:

Mean Squared Error (MSE)
├─ Average of (actual - predicted)²
├─ Penalizes large errors more
├─ Use when: Want to punish large errors
└─ Lower is better

Root Mean Squared Error (RMSE)
├─ √MSE
├─ Same units as target variable
├─ Interpretable
└─ Lower is better

Mean Absolute Error (MAE)
├─ Average of |actual - predicted|
├─ Robust to outliers
├─ Use when: Outliers present
└─ Lower is better

R² Score (Coefficient of Determination)
├─ Proportion of variance explained by model
├─ Range: -∞ to 1 (1 is perfect)
├─ Interpretation:
│  ├─ >0.9: Excellent
│  ├─ >0.7: Good
│  ├─ >0.5: Moderate
│  ├─ <0.5: Poor
│  └─ <0.0: Model worse than predicting mean
└─ Scale-independent (dimensionless)

"""


# ============================================================================
# COMMON PITFALLS & SOLUTIONS
# ============================================================================

COMMON_PITFALLS = """
╔════════════════════════════════════════════════════════════════════════╗
║                    COMMON PITFALLS & SOLUTIONS                        ║
╚════════════════════════════════════════════════════════════════════════╝

1. DATA LEAKAGE
   Problem: Information from outside training data leaks into training
   Solution: 
   └─ Scale ONLY on training data, then transform test data
   └─ Create preprocessing pipelines
   └─ Ensure all features are available at prediction time

2. OVERFITTING
   Problem: Model memorizes training data, poor test performance
   Symptoms:
   └─ High training accuracy, low test accuracy
   └─ Wide gap between train and test metrics
   Solutions:
   └─ Use more training data
   └─ Regularization (L1, L2)
   └─ Reduce model complexity
   └─ Early stopping
   └─ Cross-validation

3. UNDERFITTING
   Problem: Model too simple, captures little pattern
   Symptoms:
   └─ Low accuracy on both train and test
   Solutions:
   └─ Use more complex model
   └─ Add more features
   └─ Train longer
   └─ Reduce regularization

4. IMBALANCED CLASSES
   Problem: One class much rarer than other
   Symptoms:
   └─ High accuracy but poor performance on rare class
   └─ Misleading metrics
   Solutions:
   └─ Use stratified train-test split
   └─ Use appropriate metrics (F1, ROC AUC, not accuracy)
   └─ Class weights or resampling
   └─ Separate models for each class

5. MISSING VALUES
   Problem: Incomplete data
   Solutions:
   └─ Drop rows with missing values (if few)
   └─ Fill with mean/median/mode
   └─ Forward/backward fill (time series)
   └─ Interpolation
   └─ Create missing indicator feature

6. OUTLIERS
   Problem: Extreme values skew model
   Solutions:
   └─ Visualize to identify
   └─ Use IQR method for detection
   └─ Remove (if errors)
   └─ Cap at quantiles
   └─ Use robust algorithms (Random Forest, Median-based)

7. FEATURE SCALING
   Problem: Features on different scales
   Solutions:
   └─ StandardScaler: (x - mean) / std
   └─ MinMaxScaler: (x - min) / (max - min)
   └─ Only scale training data, apply to test
   └─ Not needed for tree-based models

8. HYPERPARAMETER TUNING
   Problem: Spending too much time searching
   Solutions:
   └─ Use RandomizedSearchCV for large spaces
   └─ Start with defaults, then refine
   └─ Use validation curves to guide search
   └─ Set reasonable parameter bounds

9. USING TEST DATA DURING TRAINING
   Problem: Inflated performance metrics
   Solutions:
   └─ NEVER look at test data during tuning
   └─ Use cross-validation on training data only
   └─ Reserve test set for final evaluation
   └─ Create separate validation set if needed

10. REPRODUCIBILITY
    Problem: Can't reproduce results
    Solutions:
    └─ Set random_state in all sklearn functions
    └─ Document data preprocessing steps
    └─ Version control code and data
    └─ Keep experiment logs

"""


# ============================================================================
# QUICK START TEMPLATE
# ============================================================================

QUICK_START = """
╔════════════════════════════════════════════════════════════════════════╗
║                    QUICK START CODE TEMPLATE                          ║
╚════════════════════════════════════════════════════════════════════════╝

# 1. IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. LOAD DATA
df = pd.read_csv('data.csv')

# 3. EXPLORATORY ANALYSIS
print(df.head())
print(df.describe())
print(df.isnull().sum())

# 4. PREPARE DATA
X = df.drop('target', axis=1)  # Features
y = df['target']                 # Target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. SCALE FEATURES
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. TRAIN MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 7. EVALUATE
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 8. FEATURE IMPORTANCE
importances = model.feature_importances_
feature_names = X.columns
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# 9. SAVE MODEL
import joblib
joblib.dump(model, 'model.pkl')

# 10. LOAD LATER
model_loaded = joblib.load('model.pkl')

"""


# ============================================================================
# LEARNING RESOURCES
# ============================================================================

RESOURCES = """
╔════════════════════════════════════════════════════════════════════════╗
║                      LEARNING RESOURCES                               ║
╚════════════════════════════════════════════════════════════════════════╝

PYTHON LIBRARIES:
├─ NumPy: Numerical computing
├─ Pandas: Data manipulation and analysis
├─ Scikit-learn: Machine learning algorithms
├─ Matplotlib: Data visualization
├─ Seaborn: Statistical data visualization
├─ TensorFlow: Deep learning
├─ PyTorch: Deep learning
└─ Keras: High-level neural networks API

DOCUMENTATION:
├─ Scikit-learn: https://scikit-learn.org/
├─ Pandas: https://pandas.pydata.org/
├─ NumPy: https://numpy.org/
└─ Matplotlib: https://matplotlib.org/

PRACTICE:
├─ Kaggle: https://www.kaggle.com/
├─ UCI ML Repository: https://archive.ics.uci.edu/
├─ Google Dataset Search: https://datasetsearch.research.google.com/
└─ GitHub repositories with datasets

KEY PAPERS:
├─ "A Few Useful Things to Know about Machine Learning" (Pedro Domingos)
├─ "No Free Lunch Theorems for Optimization" (Wolpert & Macready)
├─ "Understanding Deep Learning" (Goodfellow, Bengio, Courville)
└─ Classic papers on specific algorithms

COMMUNITIES:
├─ Stack Overflow: Ask questions
├─ r/MachineLearning: Reddit community
├─ Kaggle: Competitions and discussions
└─ ArXiv: Latest research papers

"""


# ============================================================================
# PRINT ALL GUIDES
# ============================================================================

if __name__ == "__main__":
    print(COMPLETE_ML_WORKFLOW)
    print(KEY_CONCEPTS)
    print(ALGORITHMS_GUIDE)
    print(METRICS_GUIDE)
    print(COMMON_PITFALLS)
    print(QUICK_START)
    print(RESOURCES)
    
    # Save all guides to file
    with open('e:\\Python_for_DS\\ML_REFERENCE_GUIDE.txt', 'w') as f:
        f.write(COMPLETE_ML_WORKFLOW)
        f.write(KEY_CONCEPTS)
        f.write(ALGORITHMS_GUIDE)
        f.write(METRICS_GUIDE)
        f.write(COMMON_PITFALLS)
        f.write(QUICK_START)
        f.write(RESOURCES)
    
    print("\n" + "="*70)
    print("✓ Complete ML Reference Guide saved to 'ML_REFERENCE_GUIDE.txt'")
    print("="*70)

print("\n" + "="*70)
print("YOUR MACHINE LEARNING LEARNING PATH")
print("="*70)
print("""
File Sequence for Learning:
1. 01_numpy_basics.py          → Numerical computing fundamentals
2. 02_pandas_basics.py         → Data manipulation essentials
3. 03_data_loading_exploration.py → Load and explore data
4. 04_data_preprocessing.py    → Clean and prepare data
5. 05_exploratory_data_analysis.py → Analyze and visualize
6. 06_feature_engineering.py   → Create powerful features
7. 07_sklearn_fundamentals.py  → Introduction to scikit-learn
8. 08_model_training.py        → Train and tune models
9. 09_model_evaluation.py      → Evaluate model performance
10. 10_quick_reference.py      → Quick reference guide

NEXT STEPS:
✓ Run each file to see output and learn from examples
✓ Modify code to experiment
✓ Apply to your own dataset
✓ Build end-to-end ML projects
✓ Practice, practice, practice!

Remember: The best way to learn ML is by DOING!
""")

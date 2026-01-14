import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.externals import joblib

# Comprehensive scikit-learn example
import matplotlib.pyplot as plt



# 1. Loading and preparing data
print("1. Data Loading and Preparation")
iris = load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# 2. Data Preprocessing
print("\n2. Data Preprocessing")
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training and Comparison
print("\n3. Training Multiple Models")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Compute cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.4f}")

# 4. Visualization
print("\n4. Visualizations")

# Plot model comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.title('Model Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Feature importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# 5. Making new predictions
print("\n5. Making New Predictions")
# Example of using the best model for new predictions
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sample iris measurements
sample_data_scaled = scaler.transform(sample_data)
prediction = rf_model.predict(sample_data_scaled)
print(f"Predicted class for sample data: {iris.target_names[prediction[0]]}")

# 6. Model Persistence
print("\n6. Model Persistence")

# Save the model
joblib.dump(rf_model, 'random_forest_model.joblib')

# Load the model
loaded_model = joblib.load('random_forest_model.joblib')

print("\nThis example demonstrates:")
print("- Data loading and preprocessing")
print("- Model training and evaluation")
print("- Cross-validation")
print("- Multiple model comparison")
print("- Feature importance analysis")
print("- Model persistence")
print("- Making predictions with saved models")
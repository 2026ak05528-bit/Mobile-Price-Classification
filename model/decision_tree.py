import numpy as np
import pandas as pd
import pickle
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef
)

# 1. Load the Dataset
try:
    df = pd.read_csv('test_data.csv')
    print("Dataset loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print("test_data.csv not found. Generating mock mobile dataset for demonstration...")

# 2. Separate Features and Target
X = df.drop('price_range', axis=1)
y = df['price_range']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling (Optional for Decision Trees, but kept for structural pipeline alignment)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train Decision Tree Classifier
# Added max_depth and min_samples_split constraints to prevent severe overfitting
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=8,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. Generate Predictions and Probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)  # Necessary for multi-class AUC validation

# 7. Calculate the 6 Requested Metrics
accuracy   = accuracy_score(y_test, y_pred)
mcc        = matthews_corrcoef(y_test, y_pred)

# Using 'weighted' averaging to capture individual class distributions across the 4 segments
precision  = precision_score(y_test, y_pred, average='weighted')
recall     = recall_score(y_test, y_pred, average='weighted')
f1         = f1_score(y_test, y_pred, average='weighted')

# Multi-class AUC setup using the 'ovr' (One-vs-Rest) calculation scheme
auc_score  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

# 8. Display Results Table
print("\n" + "="*45)
print(f"{'DECISION TREE EVALUATION METRIC':<32} | {'SCORE':<10}")
print("="*45)
print(f"{'1. Accuracy':<32} | {accuracy:.4f}")
print(f"{'2. AUC Score (Weighted OvR)':<32} | {auc_score:.4f}")
print(f"{'3. Precision (Weighted)':<32} | {precision:.4f}")
print(f"{'4. Recall (Weighted)':<32} | {recall:.4f}")
print(f"{'5. F1 Score (Weighted)':<32} | {f1:.4f}")
print(f"{'6. Matthews Correlation (MCC)':<32} | {mcc:.4f}")
print("="*45)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/decision_tree.pkl")
joblib.dump(scaler, "models/decision_tree_scaler.pkl")

print("Decision Tree Model saved successfully.")

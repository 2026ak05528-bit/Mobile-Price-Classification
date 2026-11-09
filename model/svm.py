import numpy as np
import pandas as pd
import pickle
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# 4. Feature Scaling (Mandatory for SVM Distance Maximisation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train Support Vector Classifier (SVC)
# probability=True allows the model to use Platt scaling for internal probability mapping
model = SVC(
    kernel='rbf',        # Radial Basis Function (Non-linear kernel)
    C=1.0,               # Regularization parameter
    gamma='scale',       # Kernel coefficient 
    probability=True,    # MANDATORY to compute AUC score
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. Generate Predictions and Probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)  # Enabled by probability=True above

# 7. Calculate the 6 Requested Metrics
accuracy   = accuracy_score(y_test, y_pred)
mcc        = matthews_corrcoef(y_test, y_pred)

# Using 'weighted' averaging to account for individual class sample support
precision  = precision_score(y_test, y_pred, average='weighted')
recall     = recall_score(y_test, y_pred, average='weighted')
f1         = f1_score(y_test, y_pred, average='weighted')

# Multi-class AUC calculated using the One-vs-Rest ('ovr') method
auc_score  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

# 8. Display Results Table
print("\n" + "="*45)
print(f"{'SVM EVALUATION METRIC':<32} | {'SCORE':<10}")
print("="*45)
print(f"{'1. Accuracy':<32} | {accuracy:.4f}")
print(f"{'2. AUC Score (Weighted OvR)':<32} | {auc_score:.4f}")
print(f"{'3. Precision (Weighted)':<32} | {precision:.4f}")
print(f"{'4. Recall (Weighted)':<32} | {recall:.4f}")
print(f"{'5. F1 Score (Weighted)':<32} | {f1:.4f}")
print(f"{'6. Matthews Correlation (MCC)':<32} | {mcc:.4f}")
print("="*45)

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/svm.pkl")
joblib.dump(scaler, "models/svm_scaler.pkl")

print("SVM Model saved successfully.")
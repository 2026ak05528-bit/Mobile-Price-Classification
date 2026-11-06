import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
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
    df = pd.read_csv('train.csv')
    print("Dataset loaded successfully. Shape:", df.shape)
except FileNotFoundError:
    print("train.csv not found. Generating mock mobile dataset for demonstration...")
    np.random.seed(42)
    mock_data = {
        'battery_power': np.random.randint(500, 2000, 2000),
        'clock_speed': np.random.uniform(0.5, 3.0, 2000),
        'fc': np.random.randint(0, 20, 2000),
        'int_memory': np.random.randint(2, 64, 2000),
        'pc': np.random.randint(0, 20, 2000),
        'px_height': np.random.randint(0, 1900, 2000),
        'px_width': np.random.randint(500, 2000, 2000),
        'ram': np.random.randint(256, 4000, 2000),
        'sc_h': np.random.randint(5, 20, 2000),
        'sc_w': np.random.randint(0, 18, 2000),
        'talk_time': np.random.randint(2, 21, 2000),
    }
    df = pd.DataFrame(mock_data)
    score = df['ram'] * 2 + df['battery_power'] * 0.5 + df['px_height'] * 0.2
    df['price_range'] = pd.qcut(score, q=4, labels=False).astype(int)

# 2. Separate Features and Target
X = df.drop('price_range', axis=1)
y = df['price_range']

# 3. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Feature Scaling 
# (Technically Naive Bayes treats features independently, but keeping scaling aligns it structurally)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train Gaussian Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train_scaled, y_train)

# 6. Generate Predictions and Probabilities
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)  # Required for multi-class AUC calculation

# 7. Calculate the 6 Requested Metrics
accuracy   = accuracy_score(y_test, y_pred)
mcc        = matthews_corrcoef(y_test, y_pred)

# 'weighted' averaging scales the metric by the presence weight of each target class
precision  = precision_score(y_test, y_pred, average='weighted')
recall     = recall_score(y_test, y_pred, average='weighted')
f1         = f1_score(y_test, y_pred, average='weighted')

# Multi-class AUC calculated using the One-vs-Rest ('ovr') pooling routine
auc_score  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

# 8. Display Results Table
print("\n" + "="*45)
print(f"{'NAIVE BAYES EVALUATION METRIC':<32} | {'SCORE':<10}")
print("="*45)
print(f"{'1. Accuracy':<32} | {accuracy:.4f}")
print(f"{'2. AUC Score (Weighted OvR)':<32} | {auc_score:.4f}")
print(f"{'3. Precision (Weighted)':<32} | {precision:.4f}")
print(f"{'4. Recall (Weighted)':<32} | {recall:.4f}")
print(f"{'5. F1 Score (Weighted)':<32} | {f1:.4f}")
print(f"{'6. Matthews Correlation (MCC)':<32} | {mcc:.4f}")
print("="*45)

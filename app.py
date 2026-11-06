import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, roc_auc_score, 
    confusion_matrix, classification_report
)

# Set page title and layout
st.set_page_config(page_title="Model Evaluation Dashboard", layout="wide")

st.title("📱 Mobile Price Class - Evaluation Dashboard")
st.write("Upload your test dataset to evaluate and compare pre-trained machine learning models.")

# -----------------------------------------------------------------------------
# 1. SIDEBAR: Model Selection & File Upload
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Upload & Configuration")

# Dataset upload option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

# Model selection dropdown
model_option = st.sidebar.selectbox(
    "Select ML Model",
    (
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier",
        "Support Vector Machine"
    )
)

# Mapping human-readable names to saved model filenames
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree Classifier": "decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "knn.pkl",
    "Naive Bayes Classifier": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

# -----------------------------------------------------------------------------
# Helper Function: Load Models Safely
# -----------------------------------------------------------------------------
def load_model_artifact(filename):
    """Loads a saved pickle file from the current directory."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# -----------------------------------------------------------------------------
# Main App Execution Logic
# -----------------------------------------------------------------------------
if uploaded_file is not None:
    # Read the uploaded test dataframe
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📊 Test Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Assuming 'price_range' is the target column based on the dataset structure
    target_col = 'price_range'
    
    if target_col in df.columns:
        # Separate features and target
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
        
        # Load the corresponding model
        model_filename = MODEL_FILES[model_option]
        model = load_model_artifact(model_filename)
        
        if model is not None:
            # Feature scaling setup (Applies to models requiring standardisation)
            if model_option in ["Logistic Regression", "K-Nearest Neighbor Classifier", "Support Vector Machine"]:
                scaler = load_model_artifact("scaler.pkl")
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    st.warning("⚠️ Scaler file (scaler.pkl) not found! Falling back to raw features.")
                    X_test_scaled = X_test
            else:
                X_test_scaled = X_test  # Trees/Naive Bayes don't strictly require scaling

            # Generate Predictions without retraining
            y_pred = model.predict(X_test_scaled)
            
            # Extract probability distributions for AUC calculations
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_scaled)
            else:
                # Fallback for models without native probability estimation
                y_prob = None

            # -----------------------------------------------------------------
            # c. Display of Evaluation Metrics
            # -----------------------------------------------------------------
            st.success(f"🎉 Analysis Complete for: *{model_option}*")
            
            st.subheader("📈 Multi-Class Performance Metrics")
            
            # Compute the 6 standard classification metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            mcc = matthews_corrcoef(y_test, y_pred)
            
            if y_prob is not None:
                auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
            else:
                auc = "N/A"

            # Create visual layout columns for the metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy Score", f"{acc:.4f}")
            col2.metric("AUC Score (OVR)", f"{auc:.4f}" if isinstance(auc, float) else auc)
            col3.metric("Precision (Macro)", f"{prec:.4f}")
            
            col4, col5, col6 = st.columns(3)
            col4.metric("Recall (Macro)", f"{rec:.4f}")
            col5.metric("F1-Score (Macro)", f"{f1:.4f}")
            col6.metric("Matthews Correlation (MCC)", f"{mcc:.4f}")

            # -----------------------------------------------------------------
            # d. Confusion Matrix & Classification Report
            # -----------------------------------------------------------------
            st.markdown("---")
            rep_col, cm_col = st.columns([1, 1])
            
            with rep_col:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4), use_container_width=True)
                
            with cm_col:
                st.subheader("🧩 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                # Display standard confusion matrix as a formatted dataframe
                cm_df = pd.DataFrame(
                    cm, 
                    index=[f"Actual {i}" for i in range(cm.shape[0])],
                    columns=[f"Predicted {i}" for i in range(cm.shape[1])]
                )
                st.dataframe(cm_df, use_container_width=True)

        else:
            st.error(f"❌ Missing artifact: Could not find pre-trained file {model_filename} in the working directory.")
    else:
        st.error(f"❌ Column schema failure: Target column {target_col} is missing from the uploaded file.")
else:
    st.info("💡 Awaiting file submission. Drop your evaluation test CSV in the sidebar menu to begin.")
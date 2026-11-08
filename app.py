import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Model Evaluation Dashboard",
    layout="wide"
)

st.title("📱 Mobile Price Classification Dashboard")
st.markdown(
    "Upload a test dataset and evaluate pre-trained machine learning models."
)

# -----------------------------------------------------------------------------
# Model Files
# -----------------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree Classifier": "models/decision_tree.pkl",
    "K-Nearest Neighbor Classifier": "models/knn.pkl",
    "Naive Bayes Classifier": "models/naive_bayes.pkl",
    "Support Vector Machine": "models/svm.pkl"
}

# -----------------------------------------------------------------------------
# Helper Function
# -----------------------------------------------------------------------------
def load_model_artifact(filename):
    try:
        return joblib.load(filename)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Upload Section
# -----------------------------------------------------------------------------
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    with st.container(border=True):

        st.subheader("📁 Upload & Configuration")

        uploaded_file = st.file_uploader(
            "Upload Test CSV File",
            type=["csv"]
        )

        selected_model = st.selectbox(
            "Select ML Model",
            list(MODEL_FILES.keys())
        )

# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------
if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        target_col = "price_range"

        if target_col not in df.columns:
            st.error(
                f"❌ Target column '{target_col}' not found in uploaded dataset."
            )
            st.stop()

        # ---------------------------------------------------------------------
        # Tabs
        # ---------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs(
            [
                "📈 Evaluation Results",
                "📋 Complete Dataset",
                "📊 Model Comparison",
            ]
        )

        # ---------------------------------------------------------------------
        # Dataset Preparation
        # ---------------------------------------------------------------------
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]

        # =====================================================================
        # TAB 1 : MODEL EVALUATION
        # =====================================================================
        with tab1:

            st.subheader("📈 Selected Model Evaluation")

            model_filename = MODEL_FILES[selected_model]
            model = load_model_artifact(model_filename)

            if model is None:
                st.error(
                    f"❌ Could not load model file: {model_filename}"
                )
                st.stop()

            # Scaling for selected models
            if selected_model in [
                "Logistic Regression",
                "Decision Tree Classifier",
                "K-Nearest Neighbor Classifier",
                "Naive Bayes Classifier",
                "Support Vector Machine"
            ]:

                MODEL_SCALERS = {
                    "Logistic Regression":"models/logistic_scaler.pkl",
                    "Decision Tree Classifier":"models/decision_tree_scaler.pkl",
                    "K-Nearest Neighbor Classifier":"models/knn_scaler.pkl",
                    "Naive Bayes Classifier": "models/naive_bayes_scaler.pkl",
                    "Support Vector Machine":"models/svm_scaler.pkl"
                }
                
                scaler = load_model_artifact(MODEL_SCALERS[selected_model])

                if scaler is not None:
                    X_eval = scaler.transform(X_test)
                else:
                    st.warning(
                        "⚠️ scaler.pkl not found. Using raw features."
                    )
                    X_eval = X_test

            else:
                X_eval = X_test

            # Predictions
            y_pred = model.predict(X_eval)

            # Probabilities
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)
            else:
                y_prob = None

            # Metrics
            acc = accuracy_score(y_test, y_pred)

            prec = precision_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            rec = recall_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            mcc = matthews_corrcoef(
                y_test,
                y_pred
            )

            try:
                if y_prob is not None:
                    auc = roc_auc_score(
                        y_test,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                    )
                else:
                    auc = None
            except Exception:
                auc = None

            st.success(
                f"🎉 Evaluation Complete for {selected_model}"
            )

            # Metric Cards
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Accuracy",
                f"{acc:.4f}"
            )

            col2.metric(
                "AUC",
                f"{auc:.4f}" if auc is not None else "N/A"
            )

            col3.metric(
                "Precision",
                f"{prec:.4f}"
            )

            col4, col5, col6 = st.columns(3)

            col4.metric(
                "Recall",
                f"{rec:.4f}"
            )

            col5.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            col6.metric(
                "MCC",
                f"{mcc:.4f}"
            )

            st.markdown("---")

            report_col, cm_col = st.columns(2)

            with report_col:

                st.subheader("📋 Classification Report")

                report_df = pd.DataFrame(
                    classification_report(
                        y_test,
                        y_pred,
                        output_dict=True,
                        zero_division=0,
                    )
                ).transpose()

                st.dataframe(
                    report_df,
                    use_container_width=True,
                )

            with cm_col:

                st.subheader("🧩 Confusion Matrix")

                cm = confusion_matrix(
                    y_test,
                    y_pred
                )

                cm_df = pd.DataFrame(
                    cm,
                    index=[
                        f"Actual {i}"
                        for i in range(cm.shape[0])
                    ],
                    columns=[
                        f"Predicted {i}"
                        for i in range(cm.shape[1])
                    ],
                )

                st.dataframe(
                    cm_df,
                    use_container_width=True,
                )

        # =====================================================================
        # TAB 2 : COMPLETE DATASET
        # =====================================================================
        with tab2:

            st.subheader("📋 Complete Uploaded Dataset")

            st.write(
                f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
            )

            st.dataframe(
                df,
                use_container_width=True,
                height=600,
            )

        # =====================================================================
        # TAB 3 : MODEL COMPARISON
        # =====================================================================
        with tab3:

            st.subheader("📊 Comparison of All Models")

            comparison_results = []

            for model_name, model_path in MODEL_FILES.items():

                try:

                    current_model = load_model_artifact(
                        model_path
                    )

                    if current_model is None:
                        continue

                    # Scaling
                    if model_name in [
                        "Logistic Regression",
                        "Decision Tree Classifier",
                        "K-Nearest Neighbor Classifier",
                        "Naive Bayes Classifier",
                        "Support Vector Machine"
                    ]:

                        MODEL_SCALERS = {
                            "Logistic Regression":"models/logistic_scaler.pkl",
                            "Decision Tree Classifier":"models/decision_tree_scaler.pkl",
                            "K-Nearest Neighbor Classifier":"models/knn_scaler.pkl",
                            "Naive Bayes Classifier":"models/naive_bayes_scaler.pkl",
                            "Support Vector Machine":"models/svm_scaler.pkl"
                        }

                        scaler = load_model_artifact(
                            MODEL_SCALERS[model_name]
                        )

                        if scaler is not None:
                            X_model = scaler.transform(X_test)
                        else:
                            X_model = X_test

                    else:
                        X_model = X_test

                    predictions = current_model.predict(
                        X_model
                    )

                    accuracy = accuracy_score(
                        y_test,
                        predictions
                    )

                    precision = precision_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )

                    recall = recall_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )

                    f1score = f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )

                    mcc_score = matthews_corrcoef(
                        y_test,
                        predictions
                    )

                    auc_score = np.nan

                    if hasattr(
                        current_model,
                        "predict_proba"
                    ):

                        try:

                            probabilities = (
                                current_model.predict_proba(
                                    X_model
                                )
                            )

                            auc_score = roc_auc_score(
                                y_test,
                                probabilities,
                                multi_class="ovr",
                                average="macro",
                            )

                        except Exception:
                            pass

                    comparison_results.append(
                        {
                            "Model": model_name,
                            "Accuracy": round(
                                accuracy, 4
                            ),
                            "Precision": round(
                                precision, 4
                            ),
                            "Recall": round(
                                recall, 4
                            ),
                            "F1 Score": round(
                                f1score, 4
                            ),
                            "MCC": round(
                                mcc_score, 4
                            ),
                            "AUC": (
                                round(auc_score, 4)
                                if not pd.isna(
                                    auc_score
                                )
                                else "N/A"
                            ),
                        }
                    )

                except Exception as e:

                    st.warning(
                        f"Error evaluating {model_name}: {e}"
                    )

            comparison_df = pd.DataFrame(
                comparison_results
            )

								

														
						 
																			
				 

					 
															  
			 

            if not comparison_df.empty:

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )

                numeric_df = comparison_df.copy()

                numeric_df["Accuracy"] = pd.to_numeric(
                    numeric_df["Accuracy"]
                )

                best_model = numeric_df.loc[
                    numeric_df["Accuracy"].idxmax()
                ]

                st.success(
                    f"""
                    🏆 Best Performing Model: {best_model['Model']}
                    
                    Accuracy: {best_model['Accuracy']}
                    """
                )

            else:
                st.warning(
                    "No models available for comparison."
                )

    except Exception as e:
        st.error(f"❌ Error while processing file: {e}")

else:
    st.info(
        "💡 Upload a test CSV file to begin evaluation."
    )
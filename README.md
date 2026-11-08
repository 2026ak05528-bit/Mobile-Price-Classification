Mobile Price Classification
A. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models capable of predicting the price category of a mobile phone based on its hardware specifications and features.

The project implements the following classification algorithms:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Naive Bayes Classifier
Support Vector Machine (SVM)

The models are evaluated using multiple performance metrics and deployed using Streamlit for interactive prediction and analysis.

B. Dataset Description
Dataset Name

Mobile Price Classification Dataset

Dataset Source

Kaggle: Mobile Price Classification Dataset

Dataset Characteristics
Total Instances: 2000
Features: 20 input features + 1 target variable
Classification Type: Multi-Class Classification
Feature Description
Feature	Description
battery_power	Total battery capacity
blue	Bluetooth support
clock_speed	Processor speed
dual_sim	Dual SIM support
fc	Front camera megapixels
four_g	4G support
int_memory	Internal memory
m_dep	Mobile depth
mobile_wt	Phone weight
n_cores	Number of CPU cores
pc	Primary camera megapixels
px_height	Pixel resolution height
px_width	Pixel resolution width
ram	RAM size
sc_h	Screen height
sc_w	Screen width
talk_time	Battery talk time
three_g	3G support
touch_screen	Touch screen support
wifi	Wi-Fi support
Target Variable
Class	Price Category
0	Low Cost
1	Medium Cost
2	High Cost
3	Very High Cost

The goal is to predict the correct mobile phone price category using the above hardware specifications.

C. GitHub Repository Link

GitHub Repository

Plain Text
https://github.com/2026ak05528-bit/Mobile-Price-Classification

Live Streamlit App

Plain Text
https://your-app-link.streamlit.app

Replace the Streamlit URL with your deployed application link.

## Project Structure

```text
Mobile-Price-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── scaler.pkl

---

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib
- OS
- Pickle

---

## D. Models Used

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score | MCC | AUC |
|--------|----------|-----------|--------|----------|------|------|
| Logistic Regression | 0.9745 | 0.9745 | 0.9745 | 0.9744 | 0.9661 | 0.9991 |
| Decision Tree | 0.9495 | 0.9499 | 0.9495 | 0.9496 | 0.9327 | 0.9838 |
| KNN | 0.6560 | 0.6621 | 0.6560 | 0.6568 | 0.5423 | 0.8659 |
| Naive Bayes | 0.8170 | 0.8178 | 0.8170 | 0.8173 | 0.7560 | 0.9544 |
| Support Vector Machine | 0.9650 | 0.9653 | 0.9650 | 0.9651 | 0.9534 | 0.9978 | 

Model Performance Observations
Logistic Regression
Achieved the highest overall performance. The dataset exhibits strong linear separability, allowing Logistic Regression to classify mobile price categories effectively.

Decision Tree
Produced strong results and handled non-linear relationships effectively but slightly underperformed compared to Logistic Regression.

K-Nearest Neighbors (KNN)
Performance was lower because distance-based algorithms are sensitive to high-dimensional feature spaces.

Naive Bayes
Fast and computationally efficient. Performance was moderate due to the independence assumption between features.

Support Vector Machine (SVM)
Produced performance very close to Logistic Regression and created robust decision boundaries after feature scaling.

Overall Winner
Logistic Regression achieved the best overall results with the highest Accuracy, MCC, and AUC scores.

Test Data
The repository contains:

test_data.csv
The file contains:

All required input features
Target column (price_range)
Unseen test samples used for evaluation
The deployed Streamlit application loads pre-trained models and evaluates the uploaded test dataset without retraining.

Streamlit Features
The Streamlit application provides:

✅ Test CSV Upload
✅ Model Selection Dropdown
✅ Classification Report
✅ Evaluation Metrics
✅ Confusion Matrix
✅ Model Performance Comparison
✅ Prediction Results
✅ Support for All Five Classification Models
These features satisfy the assignment requirements.

Requirements
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
How to Run
pip install -r requirements.txt
streamlit run app.py

Author
KONTHAM SAIDATTA SOUJANYA
M.Tech. Artificial Intelligence & Machine Learning
BITS Pilani WILP Program

**a. Problem statement:** The objective of this project is to build and compare multiple machine learning classification models capable of predicting the price category of a mobile phone based on its hardware specifications and features.
The project implements five classification algorithms:
    Logistic Regression
    Decision Tree Classifier
    K-Nearest Neighbors (KNN)
    Naive Bayes Classifier
    Support Vector Machine (SVM)
The models are evaluated using multiple performance metrics and deployed through a Streamlit web application for interactive prediction and analysis.

**b. Dataset description** 
Dataset Name: Mobile Price Classification Dataset
Dataset Source: Kaggle (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?resource=download)
Dataset Characteristics
_Total Instances:_ 2000 records
_Features:_ 20 input features + 1 target variable (price_range)
_Classification Type:_ Multi-Class Classification
    Input Features
    Feature	        Description
    battery_power	Total battery capacity
    blue	        Bluetooth support
    clock_speed	    Processor speed
    dual_sim	    Dual SIM support
    fc	            Front camera megapixels
    four_g	        4G support
    int_memory	    Internal memory
    m_dep	        Mobile depth
    mobile_wt	    Phone weight
    n_cores	        Number of cores
    pc	            Primary camera megapixels
    px_height	    Pixel resolution height
    px_width	    Pixel resolution width
    ram	            RAM size
    sc_h	        Screen height
    sc_w	        Screen width
    talk_time	    Battery talk time
    three_g	        3G support
    touch_screen	Touch screen support
    wifi	        WiFi support
    
_Target Variable:_ price_range
    Class	Meaning
    0	    Low Cost
    1	    Medium Cost
    2	    High Cost
    3	    Very High Cost
    
The goal is to predict the correct mobile phone price category using the above hardware specifications.

**c. Github Repository Link:** https://github.com/2026ak05528-bit/Mobile-Price-Classification.git

_Git folder structure_
project-folder/
│
├── app.py - the Streamlit app, which loads the saved models and never retarins
├── requirements.txt - list of softwares used for the project execution
├── test_data.csv - the dataset used for training the document
│
├── models/  - the five saved .joblib pipelines
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   └── scaler.pkl

_Technologies Used_
    Python
    Streamlit
    Scikit-Learn
    Pandas
    NumPy
    Matplotlib
    Seaborn
    Joblib
    OS
    pickle

**d. Models used:**
_Performance Comparison_
ML Model Name             Accuracy         AUC     Precision     Recall      F1          MCC
Logistic Regression        0.9745	    0.9745	    0.9745	     0.9744	    0.9661	    0.9991
Decision Tree              0.9495       0.9499	    0.9495	     0.9496	    0.9327  	0.9838
KNN                        0.656	    0.6621	    0.656	     0.6568  	0.5423	    0.8659
Naive Bayes                0.817	    0.8178	    0.817	     0.8173	    0.756    	0.9544
Support Vector Machine     0.965    	0.9653	    0.965        0.9651	    0.9534	    0.9978

_Model Performance Observations_
ML Model	                Observation
Logistic Regression	        Achieved the highest performance among all models. The features exhibit strong linear separability, allowing Logistic Regression to perform exceptionally well.
Decision Tree	            Produced strong results and handled non-linear relationships effectively but showed slightly lower generalization than Logistic Regression.
KNN	                        Performance was significantly lower. High-dimensional feature space reduced neighbor-based classification effectiveness.
Naive Bayes	                Fast and computationally efficient. Performance depends on the feature independence assumption.
Support Vector Machine	    Expected to perform competitively due to its ability to create optimal decision boundaries, especially after feature scaling.
Overall Winner	            Logistic Regression demonstrated the best overall performance based on Accuracy, F1 Score, MCC Score, and AUC Score.

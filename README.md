Cancer Detection using Machine Learning
Project Overview
This project provides a machine learning solution for cancer detection, classifying tumors as malignant or benign based on various features from cell nuclei. The core of the project involves:


Data Preparation: Loading and preprocessing the data.csv dataset, including handling imbalanced classes using SMOTE.


Model Training: Training and evaluating three distinct classification models: Random Forest, Support Vector Machine (SVC), and XGBoost.


Model Evaluation: Presenting key performance metrics like accuracy, confusion matrix, and classification report for each model.


Model Saving: Saving trained models for future predictions.

Dataset
The data.csv file contains features derived from digitized images of breast mass FNA. It includes 

id, diagnosis (target variable: M/B), and various mean, standard error, and "worst" values for cell characteristics (e.g., radius_mean, texture_mean, perimeter_mean).

Getting Started
Prerequisites
Python 3.x 


pandas, numpy, seaborn, matplotlib, scikit-learn, xgboost, imblearn 

Install dependencies:

Bash

pip install pandas numpy seaborn matplotlib scikit-learn xgboost imbalanced-learn
Installation
Clone the repository:

Bash

git clone https://github.com/your-username/cancer-detection.git
cd cancer-detection
Ensure 

data.csv is in the project root.

Usage
Run the main script to train and evaluate models:

Bash

python cancer_detection.py
The script will output performance metrics for each model and save the trained models as 

.pkl files.

Output Example
Model: RandomForest
Best Params: {'n_estimators': 500}
Accuracy: 0.9824561403508771
Confusion Matrix:
 [[67  2]
 [ 0 45]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.97      0.99        69
           1       0.96      1.00      0.98        45

    accuracy                           0.98       114
   macro avg       0.98      0.99      0.98       114
weighted avg       0.98      0.98      0.98       114

--------------------------------------------------
Contributing
Feel free to open issues or submit pull requests for improvements.

License
This project is open-source under the MIT License.

# Cancer Detection using Machine Learning

## Project Overview

This project implements a machine learning pipeline for cancer detection using various classification algorithms. The goal is to predict whether a tumor is malignant (M) or benign (B) based on several features extracted from cell nuclei.

The pipeline includes:
* **Data Loading and Preprocessing**: Reading the dataset and encoding the target variable.
* **Handling Imbalanced Data**: Utilizing SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
* **Model Training and Evaluation**: Training and evaluating popular machine learning models such as Random Forest, Support Vector Machine (SVC), and XGBoost using GridSearchCV for hyperparameter tuning.
* **Model Persistence**: Saving the trained models for future use.

## Dataset

[cite_start]The dataset used in this project is `data.csv`[cite: 1], which contains features computed from digitized images of fine needle aspirate (FNA) of a breast mass. Each row represents a sample, and the columns include:

* `id`: Unique identifier for the sample.
* `diagnosis`: The target variable, indicating whether the tumor is Malignant (M) or Benign (B). (This is encoded to numerical values in the script)
* `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`: These are the mean values of various characteristics of the cell nuclei.
* `radius_se`, `texture_se`, `perimeter_se`, `area_se`, `smoothness_se`, `compactness_se`, `concavity_se`, `concave points_se`, `symmetry_se`, `fractal_dimension_se`: Standard error values for the features.
* `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `smoothness_worst`, `compactness_worst`, `concavity_worst`, `concave points_worst`, `symmetry_worst`, `fractal_dimension_worst`: "Worst" or largest (mean of the three largest values) for these features.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.x installed. The following Python libraries are required:

* `pandas`
* `numpy`
* `seaborn`
* `matplotlib`
* `scikit-learn`
* `xgboost`
* `imblearn` (imbalanced-learn)

You can install these libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost imbalanced-learn
Installation
Clone the repository:

Bash

git clone [https://github.com/your-username/cancer-detection.git](https://github.com/your-username/cancer-detection.git)
cd cancer-detection
Place the dataset:
Ensure data.csv is in the root directory of the cloned repository.

Usage
To run the cancer detection pipeline and train the models, execute the cancer_detection.py script:

Bash

python cancer_detection.py
The script will perform the following steps:

Load the data.csv dataset.

Preprocess the data, converting the 'diagnosis' column into numerical format (0 for benign, 1 for malignant).

Apply SMOTE to balance the dataset.

Split the data into training and testing sets.

Train and evaluate three different models:

Random Forest Classifier

Support Vector Machine (SVC)

XGBoost Classifier

For each model, it will print:

Model name

Best hyperparameters found by GridSearchCV

Accuracy score on the test set

Confusion Matrix

Classification Report

Trained models (RandomForest.pkl, SVC.pkl, XGBoost.pkl) will be saved in the project directory.

Code Structure
cancer_detection.py: The main script containing the entire machine learning pipeline.

data.csv: The dataset used for training and testing the models.

RandomForest.pkl, SVC.pkl, XGBoost.pkl: (Generated after running the script) Pickled files of the trained models.

Results
After running cancer_detection.py, the console output will display the performance metrics for each trained model. An example output snippet for one of the models might look like this:

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
(Note: The actual accuracy and other metrics may vary slightly depending on the random_state and data split.)

Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

License
This project is open-source and available under the MIT License. 

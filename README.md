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

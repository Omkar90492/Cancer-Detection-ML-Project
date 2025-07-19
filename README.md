# 🧬 Cancer Detection ML

A machine-learning project that predicts whether a tumor is **benign** or **malignant** using clinical measurement features.

This repository includes:
- Data preprocessing with SMOTE
- Training of **Random Forest**, **SVC**, and **XGBoost**
- Evaluation reports (accuracy, precision, recall, F1-score)
- Saved models in `.pkl` format for future use

---

## ⚙️ Features

- ✅ Trains 3 robust classification models  
- ⚖️ Addresses class imbalance using **SMOTE**  
- 📊 Outputs detailed model metrics  
- 💾 Saves trained models for reuse  
- 💡 Easy to extend (e.g., add a web UI)

---

📁 Project Structure
-------------------
- cancer_detection.py — Main script
- requirements.txt — Dependencies
- /models — Trained models
- /data — Breast Cancer dataset
- /notebooks — Jupyter notebook experiments

  ---


## 🧠 Tech Stack

- **Python 3.7+** – Programming language  
- **Pandas** – Data loading and preprocessing  
- **NumPy** – Numerical operations  
- **Matplotlib** – Basic data visualization  
- **Seaborn** – Statistical plotting (optional)  
- **Scikit-learn** – Machine learning models and evaluation  
  - RandomForest, SVC, train-test split, metrics  
- **XGBoost** – Gradient boosting classifier  
- **Imbalanced-learn (SMOTE)** – Class imbalance handling  
- **Pickle** – Saving and loading trained models
  
---

## 🚀 How to Run the Project

---

Follow these steps to set up and run the project locally:

***Step 1: Clone the Repository***

git clone https://github.com/your-username/your-repo-name.git

cd your-repo-name

---

***Step 2: Install the Required Dependencies***

pip install -r requirements.txt

---

***Step 3: Run the Main Script***

python cancer_detection.py

---

***Step 4: Output***

The script will:

Preprocess the dataset

Handle class imbalance using SMOTE

Train RandomForest, SVC, and XGBoost models

Display classification reports and confusion matrices

Save models as .pkl files

---

## 📌 Results

## 📈 Accuracy Chart
![Accuracy](images/accuracy_comparison.png)

## 🧪 Confusion Matrix - XGBoost
![XGBoost](images/confusion_matrix_xgboost.png)

## 🌲 Confusion Matrix - Random Forest
![RandomForest](images/confusion_matrix_randomforest.png)

## 💻 Confusion Matrix - SVC
![SVC](images/confusion_matrix_svc.png)

---

##🚀 Quick Tips
-------------
- ✅ Use a virtual environment to avoid conflicts
- 📌 Run the notebook first if you want step-by-step exploration
- 🧠 Modify hyperparameters in `cancer_detection.py` for better accuracy
- 📈 Logs and results are saved in the `outputs/` folder

---

##✅ Prerequisites
Before running this project, ensure you have the following installed:
Python 3.7 or higher
Jupyter Notebook or any Python IDE
pip (Python package installer)

---

##🧾 Conclusion
This project demonstrates how machine learning algorithms can be used to detect cancer using a dataset of diagnostic features. By comparing different models such as Random Forest, SVC, and XGBoost, we were able to evaluate and visualize their performance through accuracy scores and confusion matrices.
✅ Key outcomes:
Successfully trained and evaluated multiple ML models.
Achieved strong prediction performance (refer to accuracy_comparison.png).
Visualized model results using confusion matrices for deeper insights.

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

Step 1: Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
---
Step 2: Install the required dependencies
pip install -r requirements.txt
---
Step 3: Run the main script
python cancer_detection.py
---
Step 4: Output
The script will:
Preprocess the dataset
Handle class imbalance using SMOTE
Train RandomForest, SVC, and XGBoost models
Display classification reports and confusion matrices
Save models as .pkl files
---

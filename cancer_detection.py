import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# 1️⃣ Load Data
data = pd.read_csv('data.csv')
data['diagnosis'] = data['diagnosis'].astype('category').cat.codes

# 2️⃣ Prepare Features (X) and Target (y)
x = data.drop(['id', 'diagnosis'], axis=1)
y = data['diagnosis']

# 3️⃣ Handle Imbalanced Data with SMOTE
sm = SMOTE(random_state=42)
X_res, Y_res = sm.fit_resample(x, y)

# 4️⃣ Split Data for Training and Testing
x_train, x_test, y_train, y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=10)

# 5️⃣ Define a Function to Train and Save Models
def FitModel(algo_name, algorithm, params):
    grid = GridSearchCV(algorithm, param_grid=params, cv=10, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(x_train, y_train)
    best_model = grid.best_estimator_
    pickle.dump(grid, open(algo_name + '.pkl', 'wb'))
    preds = grid.predict(x_test)
    print(f"Model: {algo_name}")
    print("Best Params:", grid.best_params_)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print("Classification Report:\n", classification_report(y_test, preds))
    print("-" * 50)

# 6️⃣ Train Random Forest Model
FitModel('RandomForest', RandomForestClassifier(), {'n_estimators': [500]})

# 7️⃣ Train Support Vector Machine Model
FitModel('SVC', SVC(), {'C': [1], 'gamma': [0.001]})

# 8️⃣ Train XGBoost Model
FitModel('XGBoost', XGBClassifier(), {'n_estimators': [500]})

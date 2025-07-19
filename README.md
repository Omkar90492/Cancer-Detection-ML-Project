# Cancer Detection Using Machine Learning

This project uses machine learning models to detect cancer based on medical data. It leverages Random Forest, Support Vector Machine (SVC), and XGBoost classifiers trained on a labeled dataset.

## 📂 Project Structure

├── cancer_detection.py # Main script for detection logic
├── Import Libraries.py # Script for importing required libraries
├── data.csv # Dataset used for training/testing
├── RandomForest.pkl # Trained Random Forest model
├── SVC.pkl # Trained Support Vector Classifier
├── XGBoost.pkl # Trained XGBoost model
└── .vscode/ # VS Code config files

markdown
Copy
Edit

## 📊 Dataset

The dataset (`data.csv`) contains medical parameters relevant to cancer diagnosis. Ensure this file is in the root directory before running the model.

## 💡 Features Used

The features depend on the dataset but typically include:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- ...and other statistical features used in diagnosis.

## 🧠 Machine Learning Models

The following models are trained and included:
- **Random Forest**
- **Support Vector Classifier (SVC)**
- **XGBoost**

These models are stored as `.pkl` files and can be loaded directly for prediction.

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/cancer-detection-ml.git
cd cancer-detection-ml
2. Install Dependencies
Make sure you have Python 3.7+ and install required packages:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not present, install manually:

bash
Copy
Edit
pip install pandas scikit-learn xgboost
3. Run the Detection Script
bash
Copy
Edit
python cancer_detection.py
Ensure data.csv and model .pkl files are in the same directory.

🧪 Model Evaluation
Each model was evaluated using accuracy, precision, recall, and F1-score metrics. You can expand this project by adding cross-validation and confusion matrix plots.

📈 Future Improvements
Add a web interface (e.g., Flask or Streamlit)

Enable live model training with uploaded datasets

Visualize predictions and feature importance

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss the changes.

📄 License
This project is licensed under the MIT License.

# 🚀 Equipment Failure Prediction using XGBoost and SHAP Explainability

An end-to-end Machine Learning project that predicts equipment failures using XGBoost and provides model explainability with SHAP. The project also includes an interactive Streamlit web application for real-time predictions.

---

## ✨ Features

- Predicts equipment failure using XGBoost.
- Handles imbalanced data using SMOTE.
- Provides model explainability using SHAP.
- Interactive Streamlit web application.
- Real-time prediction with user inputs.
- Clean and modular project structure.
- Easy deployment and reproducibility.

---

## 📂 Project Structure

```text
equipment-failure-prediction/
│
├── app/
│   └── app.py
│
├── models/
│   ├── xgboost_model.pkl
│   ├── best_threshold.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── reports/
│   ├── figures/
│   └── model_results.csv
│
├── src/
│   └── predict.py
│
├── requirements.txt
├── .gitignore
└── README.md
```
---

## 🛠️ Technologies Used

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalanced Data Handling | SMOTE |
| Model Explainability | SHAP |
| Web Framework | Streamlit |
| Visualization | Matplotlib |
| Model Serialization | Joblib |
| Version Control | Git & GitHub |

---

## 🔄 Machine Learning Pipeline

1. Data Collection
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Handling Class Imbalance using SMOTE
6. Train-Test Split
7. Model Training using XGBoost
8. Threshold Tuning
9. Model Evaluation
10. SHAP Explainability
11. Model Serialization using Joblib
12. Streamlit Deployment

---

## 📂 Dataset Description

The dataset contains operational and sensor-related information collected from industrial equipment. It is used to predict whether a machine is likely to fail based on its operating conditions.

### Dataset Features

- Air Temperature
- Process Temperature
- Rotational Speed (RPM)
- Torque
- Tool Wear
- Machine Type
- Failure Type

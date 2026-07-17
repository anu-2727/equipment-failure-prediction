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

---

## 📊 Model Performance

The XGBoost model achieved strong predictive performance on the equipment failure dataset.

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

The model was further optimized using threshold tuning to improve failure detection while maintaining a good balance between precision and recall.

---

## 🔍 SHAP Explainability

To improve model transparency and interpretability, SHAP (SHapley Additive exPlanations) was used.

### SHAP helps to:

- Explain individual predictions.
- Identify the most important features.
- Understand feature contributions.
- Increase trust in model predictions.
- Support better decision-making.

---

## 🌐 Streamlit Dashboard

The project includes an interactive Streamlit web application that allows users to predict equipment failures in real time.

### Dashboard Features

- User-friendly interface
- Real-time equipment failure prediction
- Easy input of machine parameters
- Fast prediction results
- Interactive and responsive design

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/anu-2727/equipment-failure-prediction.git
```

### 2. Navigate to the Project Folder

```bash
cd equipment-failure-prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

After installing the required dependencies, run the Streamlit application using the following command:

```bash
streamlit run app.py
```

Once the application starts, open the URL displayed in your terminal (usually http://localhost:8501) in your web browser.

### Using the Application

1. Enter the required equipment parameters.
2. Click the **Predict** button.
3. View the prediction result.
4. Analyze the SHAP explainability output for feature importance.

---

---

## 👩‍💻 Author

**Ponanupriya M**

- 🎓 B.Tech – Artificial Intelligence and Data Science
- 🏫 Ramco Institute of Technology
- 💻 GitHub: https://github.com/anu-2727
- 💼 LinkedIn: www.linkedin.com/in/ponanupriya-m-986221338

  
## 🚀 Future Improvements

- Deploy the application on Streamlit Community Cloud.
- Add user authentication.
- Support batch predictions using CSV upload.
- Integrate a real-time monitoring dashboard.
- Improve model performance using hyperparameter tuning.
- Containerize the application using Docker.
- Implement CI/CD using GitHub Actions.

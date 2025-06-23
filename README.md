# 🛡️ FraudShield AI – Credit Card Fraud Detection

FraudShield AI is a powerful, explainable, and interactive fraud detection dashboard built using **XGBoost**, **Random Forest**, **SHAP**, **SMOTE**, and **Streamlit**. It detects fraudulent transactions in real time while offering **visual explainability**, **voice alerts**, and **dynamic model comparison**.

---

## 🚀 Features

- 🔍 **Fraud Detection** using XGBoost & Random Forest.
- 🎛️ **Model Switcher** to choose your classifier dynamically.
- 🧠 **SHAP Explainability** to visualize feature impacts.
- 📈 **Interactive Visualizations** (Plotly, Seaborn).
- 🧪 **Unseen Dataset Testing** for real-world simulation.
- 🔊 **Voice Alerts** for real-time fraud notifications.
- 📊 **Model Comparison** with Accuracy, Precision, Recall, F1, ROC-AUC.
- 📝 **Prediction History Logging**.
- ⚖️ **SMOTE** used to handle data imbalance.

---

## 🧠 Tech Stack

| Tool / Library     | Role                             |
|--------------------|----------------------------------|
| Python             | Programming Language             |
| Pandas, NumPy      | Data Manipulation                |
| Scikit-learn       | Preprocessing, Evaluation        |
| XGBoost, RandomForest | ML Models                   |
| SHAP               | Model Explainability             |
| SMOTE              | Imbalanced Data Handling         |
| Streamlit          | Dashboard Web App                |
| Plotly, Seaborn    | Data Visualization               |

---

## 🗃️ Dataset

- `train_data.csv` – Training data (SMOTE applied).
- `test_cases.csv` – Unseen dataset for prediction simulation.

📌 Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> Features `V1` to `V28` are anonymized via PCA due to confidentiality.

---

## 📊 Model Performance (on Unseen Test Data)

| Metric      | XGBoost     | Random Forest |
|-------------|-------------|----------------|
| Accuracy    | 99.9%       | 99.7%          |
| Precision   | 96.5%       | 95.2%          |
| Recall      | 94.3%       | 92.1%          |
| F1 Score    | 95.4%       | 93.6%          |
| ROC-AUC     | 0.999       | 0.998          |

---

## 📦 Run Locally

```bash
git clone https://github.com/MahathiChinta/FraudShield-AI.git
cd FraudShield-AI
pip install -r requirements.txt
streamlit run app.py

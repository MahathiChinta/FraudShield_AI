import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, accuracy_score, 
                             precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Load training data
df = pd.read_csv("data/train_data.csv")
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# Normalize Amount using StandardScaler and save scaler
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X['Amount'].values.reshape(-1, 1))

# Save the scaler for use in Streamlit
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/amount_scaler.pkl")
print("✅ Amount scaler saved to 'models/amount_scaler.pkl'")

# Train-Test Split for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Show test class distribution
print("🔍 Test Set Class Distribution:\n", y_test.value_counts())

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"📊 After SMOTE: {y_train_sm.value_counts().to_dict()}")

# Unsupervised Anomaly Detection
iso = IsolationForest(contamination=0.001, random_state=42)
iso.fit(X_train)
iso_scores = [1 if s == -1 else 0 for s in iso.predict(X_test)]

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
lof_scores = [1 if s == -1 else 0 for s in lof.fit_predict(X_test)]

# XGBoost Classifier
model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_sm, y_train_sm)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=1)
rec = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)
roc_score = roc_auc_score(y_test, y_proba)

# Evaluation Report
print("📈 XGBoost Performance:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {acc:.4f}")
print(f"✅ Precision: {prec:.4f}")
print(f"✅ Recall: {rec:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
print(f"🎯 ROC-AUC Score: {roc_score:.4f}")

# Save model
joblib.dump(model, "models/xgboost_model.pkl")
print("✅ Model saved to 'models/xgboost_model.pkl'")

print("\n🌲 Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1)
rf_model.fit(X_train_sm, y_train_sm)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)

print("✅ Random Forest trained.")
print(f"🎯 ROC-AUC: {rf_auc:.4f}")
joblib.dump(rf_model, "models/random_forest_model.pkl")
print("✅ Random Forest model saved to 'models/random_forest_model.pkl'")

# Confusion Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/confusion_matrix.png")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_score:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('assets/roc_curve.png')
plt.show()

# SHAP Explainability
print("🔍 SHAP Explainability...")
explainer = shap.Explainer(model)
shap_values = explainer(X_test[:100])
shap.summary_plot(shap_values, X_test[:100], show=False)

# Save SHAP plot
os.makedirs("shap_outputs", exist_ok=True)
plt.tight_layout()
plt.savefig('shap_outputs/summary_plot.png')
print("✅ SHAP summary saved to 'shap_outputs/summary_plot.png'")

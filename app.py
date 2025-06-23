import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import streamlit.components.v1 as components  # For voice alerts
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="FraudShield AI", layout="wide")
# --------------------- Sidebar App Controls ---------------------
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

st.sidebar.markdown("### ⚙️ App Controls")

if st.sidebar.button("🔄 Reset Entire App"):
    st.session_state.reset_confirm = True

if st.session_state.get("reset_confirm", False):
    if st.sidebar.radio("Are you sure you want to reset?", ["No", "Yes"], index=0, key="confirm_choice") == "Yes":
        with st.spinner("Resetting app..."):
            time.sleep(1)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if st.sidebar.button("🗑️ Clear Prediction Results"):
    for key in list(st.session_state.keys()):
        if key.startswith("autofill_") or key == "prediction_history":
            del st.session_state[key]
    st.rerun()

# --------------------- Voice Alert ---------------------
def voice_alert(text):
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    components.html(js_code)

# --------------------- Load Resources ---------------------
model = {"XGBoost": joblib.load("models/xgboost_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl")
    }

scaler = joblib.load("models/amount_scaler.pkl")
df = pd.read_csv("data/test_cases.csv")
train_df = pd.read_csv("data/train_data_small.csv")
feature_columns = [col for col in df.columns if col not in ["Class", "Time"]]
df_display = df[feature_columns]

# --------------------- Header ---------------------
st.title("🛡️ FraudShield AI – Credit Card Fraud Detection")
st.markdown("Built with **XGBoost**, **Random Forest** and **SHAP** for explainable fraud detection.")

# --------------------- Summary ---------------------
st.markdown("---")
st.subheader("📋 Summary Panel")
total_txns = df.shape[0]
frauds_detected = df[df['Class'] == 1].shape[0]
detection_rate = (frauds_detected / total_txns) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", total_txns)
col2.metric("Frauds Detected", frauds_detected)
col3.metric("Detection Rate", f"{detection_rate:.2f}%")

# --------------------- Tabs ---------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Training Dataset", "🧪 Unseen Dataset", "🔍 Predict Transaction", "🧠 Explainability", "📊 Model Comparison"])

# --------------------- Tab 1: Training Overview ---------------------
with tab1:
    st.subheader("📈 Training Data Overview")
    if st.checkbox("Show full training dataset"):
        st.dataframe(train_df, use_container_width=True)
    else:
        st.dataframe(train_df.head(), use_container_width=True)

    st.markdown("### Class Distribution")
    fig = px.histogram(train_df, x="Class", title="Class Distribution", color="Class")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(train_df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------- Tab 2: Unseen Dataset ---------------------
with tab2:
    st.subheader("🧪 Unseen Test Dataset")
    st.info("⚠️ **Note:** This dataset was **not used for training**. It contains completely **unseen transactions**, meaning the model has **never seen these records** before. This helps test the model's real-world generalization ability.")
    df_copy = df.copy()
    df_copy["Prediction"] = df_copy["Class"].apply(lambda x: "🚨 Fraud" if x == 1 else "✅ Legit")
    st.dataframe(df_copy, use_container_width=True)
    st.caption("Unseen transactions flagged with 🚨 are marked as fraud.")
    st.markdown("#### ℹ️ Feature Description")
    st.info("""
    - The dataset's features (`V1` to `V28`) are the result of a **PCA (Principal Component Analysis)** transformation.
    - Their original meanings were intentionally **anonymized** to protect sensitive user and vendor information.
    - While we can't interpret them directly, the models learn patterns from them to distinguish fraud from legit transactions.
    """)
    
# --------------------- Tab 3: Prediction Panel ---------------------
with tab3:

    st.markdown("Here’s a quick comparison of typical Legitimate and Fraudulent transactions.")
    fraud_mean = df[df['Class'] == 1][feature_columns].mean()
    legit_mean = df[df['Class'] == 0][feature_columns].mean()

    diff_df = pd.DataFrame({
        "Feature": feature_columns,
        "Mean (Fraud)": fraud_mean.values,
        "Mean (Legit)": legit_mean.values,
        "Difference": (fraud_mean - legit_mean).abs().values
    }).sort_values("Difference", ascending=False)

    st.dataframe(diff_df.head(10), use_container_width=True)
    st.subheader("🧪 Real-World Prediction")
    model_choice = st.selectbox("Select Model for Prediction:", list(model.keys()))
    selected_model = model[model_choice]

    st.markdown("---")
    st.markdown("### 🔎 Enter Transaction Index for Prediction")
    idx = st.number_input("Enter transaction index (0 to {}):".format(len(df_display)-1), min_value=0, max_value=len(df_display)-1, step=1)

    if st.button("⚡ Predict Transaction"):
        sample = df_display.iloc[idx:idx+1]
        sample_scaled = sample.copy()
        sample_scaled["Amount"] = scaler.transform(sample_scaled[["Amount"]])

        with st.spinner("Predicting..."):
            progress_bar = st.progress(0, text="Processing...")
            for i in range(1, 6):
                time.sleep(0.2)
                progress_bar.progress(i * 20)

            prediction = selected_model.predict(sample_scaled)[0]
            proba = float(selected_model.predict_proba(sample_scaled)[0][1])

        st.subheader("🧾 Prediction Result")
        if prediction == 1:
            st.error(f"🚨 FRAUD detected with {proba*100:.2f}% confidence")
            st.progress(proba)
            voice_alert("Warning. Fraudulent transaction detected.")
        else:
            st.success(f"✅ Legitimate transaction with {(1 - proba)*100:.2f}% confidence")
            voice_alert("Transaction is safe and legitimate.")

        if "prediction_history" not in st.session_state:
            st.session_state.prediction_history = []

        st.session_state.prediction_history.append({
            "Model":model_choice,
            "Index": idx,
            "Prediction": "Fraud" if prediction == 1 else "Legit",
            "Confidence": f"{proba*100:.2f}%" if prediction == 1 else f"{(1 - proba)*100:.2f}%"
        })

    if "prediction_history" in st.session_state and st.session_state.prediction_history:
        st.markdown("### 📜 Prediction History")
        st.dataframe(pd.DataFrame(st.session_state.prediction_history), use_container_width=True)


    st.markdown("### 🧾 Real Transaction Examples: Legit vs Fraud")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### ✅ Legit Samples")
        st.dataframe(df[df["Class"] == 0].head(), use_container_width=True)
    with colB:
        st.markdown("#### 🔴 Fraud Samples")
        st.dataframe(df[df["Class"] == 1].head(), use_container_width=True)

# --------------------- Tab 4: SHAP Explainability ---------------------
with tab4:
    st.subheader("🧠 SHAP Explainability (XGBoost Only)")
    
    st.markdown("""SHAP (**SHapley Additive exPlanations**) helps you understand *why* a transaction was predicted as **fraud** or **legit**.
                In our case:
- We are using `TreeExplainer` with **XGBoost model** only (best supported).
- Each feature (like `Amount`, `V1`, `V2`, etc.) gets a score that shows its **impact on the prediction**.
- **Positive SHAP values** push the prediction **toward fraud**.
- **Negative SHAP values** push it **toward legit**.""")

    shap_index = st.number_input("🎯 Select index to explain (XGBoost only):", 0, len(df_display)-1, value=0, key='shap_idx')
    shap_sample = df_display.iloc[shap_index:shap_index+1]
    shap_sample_scaled = shap_sample.copy()
    amount_scaler = MinMaxScaler()
    amount_scaler.fit(df[['Amount']])
    shap_sample_scaled['Amount'] = amount_scaler.transform(shap_sample_scaled[['Amount']])

    # SHAP explanation only for XGBoost
    xgb_model = model["XGBoost"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(shap_sample_scaled)

    prediction = xgb_model.predict(shap_sample_scaled)[0]
    proba = float(xgb_model.predict_proba(shap_sample_scaled)[0][1])

    if prediction == 1:
        st.error(f"🚨 Transaction {shap_index} predicted as **FRAUD** with {proba:.2%} confidence")
    else:
        st.success(f"✅ Transaction {shap_index} predicted as **LEGIT** with {(1 - proba):.2%} confidence")

    st.subheader("🔍 SHAP Feature Importance")
    shap_df = pd.DataFrame({
        "Feature": shap_sample.columns,
        "SHAP Value": shap_values[0]
    }).sort_values("SHAP Value", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color="orange")
    ax.set_title("Top SHAP Contributions")
    ax.set_xlabel("SHAP Value")
    ax.invert_yaxis()
    st.pyplot(fig)

# --------------------- Tab 5: Model Comparison ---------------------
with tab5:
    st.subheader("📊 Model Comparison & Evaluation")
    st.markdown("This section compares the performance of **XGBoost** and **Random Forest** models using the unseen test dataset. Each model was evaluated using standard classification metrics.")

    # Define metrics
    metrics = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "ROC-AUC": []
    }

    names = ["XGBoost", "Random Forest"]

    for name in names:
        clf = model[name]
        # Scale amount feature
        X_eval = df_display.copy()
        amount_scaler = MinMaxScaler()
        amount_scaler.fit(df[["Amount"]])  # or use X_train["Amount"]
        X_eval["Amount"] = amount_scaler.transform(X_eval[["Amount"]])


        y_pred = clf.predict(X_eval)
        y_proba = clf.predict_proba(X_eval)[:, 1]

        metrics["Model"].append(name)
        metrics["Accuracy"].append(accuracy_score(df["Class"], y_pred))
        metrics["Precision"].append(precision_score(df["Class"], y_pred, zero_division=1))
        metrics["Recall"].append(recall_score(df["Class"], y_pred, zero_division=1))
        metrics["F1 Score"].append(f1_score(df["Class"], y_pred, zero_division=1))
        metrics["ROC-AUC"].append(roc_auc_score(df["Class"], y_proba))

    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Interactive bar chart
    fig = go.Figure()

    for metric in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]:
        fig.add_trace(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df[metric],
            name=metric,
            hovertemplate=f"%{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>"
        ))

    fig.update_layout(
        title="📊 Model Evaluation Metrics (Interactive)",
        xaxis_title="Model",
        yaxis_title="Score",
        barmode="group",
        height=500,
        template="plotly_white",
        legend_title="Metric"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Conclusion
    st.markdown("---")
    st.markdown("### 🏆 Model Selection Conclusion")

    if metrics_df.loc[0, "F1 Score"] > metrics_df.loc[1, "F1 Score"]:
        best_model = "XGBoost"
    else:
        best_model = "Random Forest"

    st.success(f"✅ Based on **F1 Score** and **ROC-AUC**, **{best_model}** performs better overall on the unseen dataset.")

    st.markdown(f"""
- **{best_model}** shows stronger **generalization** on unseen test data.
- It offers a better balance between **precision** (avoiding false alarms) and **recall** (catching actual frauds).
- F1 Score is a key metric in fraud detection as it considers both **false positives** and **false negatives**.

📌 Both the models were trained on the **same training data**, using **SMOTE** to handle class imbalance.
""")


# --------------------- Footer ---------------------
st.markdown("---")
st.markdown("👨‍💻 Developed by **Shanmukha Mahathi Chinta** ")

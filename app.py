import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "fraud_detection_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.error(f"❌ Model file not found: {MODEL_PATH}")
    st.stop()

st.title("💳 Fraud Detection Prediction App")
st.markdown("Enter transaction details below and click **Predict** to check for possible fraud.")

st.divider()


transaction_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER", "CASH_OUT"])
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=900.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)


if st.button("🔍 Predict"):

    
    BalanceDiffOrig = oldbalanceOrg - newbalanceOrig
    BalanceDiffDest = newbalanceDest - oldbalanceDest

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "BalanceDiffOrig": BalanceDiffOrig,
        "BalanceDiffDest": BalanceDiffDest
    }])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1] 

    if transaction_type == "PAYMENT" and oldbalanceOrg < amount:
        st.warning("⚠️ Suspicious: Sender’s balance is less than transaction amount — possible fraud.")

    if transaction_type == "CASH_OUT":
        if oldbalanceOrg < amount:
            st.warning("⚠️ Suspicious CASH_OUT: Withdrawal exceeds sender’s balance — possible fraud.")
        elif newbalanceOrig > oldbalanceOrg:
            st.warning("⚠️ Suspicious CASH_OUT: New balance cannot be greater than old balance — possible fraud.")


    if prediction == 1:
        st.error(f"🚨 **Fraudulent Transaction Detected!** (Risk Score: {prob*100:.2f}%)")
    else:
        st.success(f"✅ **Transaction Appears Safe.** (Fraud Probability: {prob*100:.2f}%)")

    DATA_FILE = "predictions_log.csv"
    input_data["prediction"] = prediction
    input_data["fraud_probability"] = prob

    if not os.path.exists(DATA_FILE):
        input_data.to_csv(DATA_FILE, index=False)
    else:
        input_data.to_csv(DATA_FILE, mode="a", header=False, index=False)

    st.info(f"📁 Data saved to `{DATA_FILE}`")

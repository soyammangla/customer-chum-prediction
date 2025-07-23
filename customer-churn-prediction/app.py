import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📱 Customer Churn Prediction App")

# Inputs
gender = st.selectbox("Gender", ["Male", "Female"])  # Just for UI
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tenure = st.slider("Tenure (in months)", 0, 72, 2)
monthly_charges = st.number_input("Monthly Charges (₹)", value=70.5)

# Encode input (without gender, as model expects only 3 features)
def encode_input():
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    return np.array([
        contract_map[contract],
        tenure,
        monthly_charges
    ]).reshape(1, -1)

# Predict
if st.button("Predict"):
    pred = model.predict(encode_input())[0]
    if pred == 1:
        st.markdown("🟡 **This customer is likely to churn.**")
    else:
        st.markdown("🟢 **This customer is not likely to churn.**")

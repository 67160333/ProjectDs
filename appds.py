import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt


# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")


# -------------------------
# Load model
# -------------------------
model = joblib.load("model_artifacts/churn_model.pkl")

with open("model_artifacts/churn_model_features.json") as f:
    feature_names = json.load(f)

with open("model_artifacts/churn_model_metadata.json") as f:
    metadata = json.load(f)


# -------------------------
# Sidebar info
# -------------------------
st.sidebar.header("Model Info")
st.sidebar.write("Model:", metadata["model"])
st.sidebar.write("Accuracy:", round(metadata["accuracy"], 3))


# -------------------------
# User input
# -------------------------
st.header("Customer Information")

credit_score = st.number_input("Credit Score", 300, 900, 600)

age = st.number_input("Age", 18, 100, 35)

tenure = st.number_input("Tenure", 0, 10, 3)

balance = st.number_input("Balance", 0.0, 300000.0, 60000.0)

products = st.selectbox("Number of Products", [1,2,3,4])

has_card = st.selectbox("Has Credit Card", [0,1])

active_member = st.selectbox("Active Member", [0,1])

salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.selectbox(
    "Geography",
    ["France","Germany","Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Female","Male"]
)


# -------------------------
# Encode categorical
# -------------------------
geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0
gender_male = 1 if gender == "Male" else 0


# -------------------------
# Predict button
# -------------------------
if st.button("Predict Churn"):

    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": has_card,
        "IsActiveMember": active_member,
        "EstimatedSalary": salary,
        "Geography_Germany": geo_germany,
        "Geography_Spain": geo_spain,
        "Gender_Male": gender_male
    }

    input_df = pd.DataFrame([input_dict])

    # match feature order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)


    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]


    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Customer likely to CHURN")
    else:
        st.success("Customer likely to STAY")


    st.write("Churn Probability:", round(prob*100,2), "%")


    # Risk level
    if prob > 0.7:
        risk = "HIGH"
    elif prob > 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.write("Risk Level:", risk)


    # Graph
    fig, ax = plt.subplots()

    labels = ["Stay","Churn"]
    values = [1-prob, prob]

    ax.bar(labels, values)
    ax.set_ylabel("Probability")
    ax.set_title("Churn Probability")

    st.pyplot(fig)

import streamlit as st
import joblib
import pandas as pd

# set the tab title
st.set_page_config(page_title="Machine Failure Prediction")

# Set the page title
st.title("Machine Failure Prediction Project")

# Set header
st.subheader("By Sakshi Chaudhari")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("preprocessing_pipeline.pkl")
model = joblib.load("gradient_boosting_model.pkl")

# Create Input boxes that take input from the user
Air_Temperature = st.number_input("Air Temperature [K]")
Process_Temperature = st.number_input("Process Temperature [K]")
Rotational_Speed = st.number_input("Rotational Speed [rpm]")
Torque = st.number_input("Torque [Nm]")
Tool_Wear = st.number_input("Tool Wear [min]")

# Include a button for prediction
submit = st.button("Predict Machine Failure")

if submit:
    data = {
        "Air temperature [K]": [Air_Temperature],
        "Process temperature [K]": [Process_Temperature],
        "Rotational speed [rpm]": [Rotational_Speed],
        "Torque [Nm]": [Torque],
        "Tool wear [min]": [Tool_Wear]
    }

    # Convert the dictionary into a DataFrame
    xnew = pd.DataFrame(data)

    # Apply data cleaning and preprocessing using the pipeline
    xnew_pre = pre.transform(xnew)

    # Make predictions
    preds = model.predict(xnew_pre)

    if preds[0] == 1:
        op = "Potential Machine Failure Detected"
    else:
        op = "No Machine Failure Predicted"

    st.subheader(op)


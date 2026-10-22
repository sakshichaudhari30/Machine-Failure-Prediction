
import streamlit as st
import joblib
import pandas as pd

# Set the tab title
st.set_page_config(page_title="Machine Failure Prediction")

# Page title
st.title("Machine Failure Prediction Project")
st.subheader("By Sakshi Chaudhari")

# Load preprocessing pipeline and model
pre = joblib.load("preprocessing_pipeline.pkl")
model = joblib.load("gradient_boosting_model.pkl")

# User inputs
Air_Temperature = st.number_input(
    "Air Temperature [K]", min_value=0.0, value=298.0
)
Process_Temperature = st.number_input(
    "Process Temperature [K]", min_value=0.0, value=308.0
)
Rotational_Speed = st.number_input(
    "Rotational Speed [rpm]", min_value=0, value=1500
)
Torque = st.number_input(
    "Torque [Nm]", min_value=0.0, value=40.0
)
Tool_Wear = st.number_input(
    "Tool Wear [min]", min_value=0, value=100
)

submit = st.button("Predict Machine Failure")

if submit:
    # Original feature names expected by the preprocessing pipeline
    data = {
        "Air temperature [K]": [Air_Temperature],
        "Process temperature [K]": [Process_Temperature],
        "Rotational speed [rpm]": [Rotational_Speed],
        "Torque [Nm]": [Torque],
        "Tool wear [min]": [Tool_Wear]
    }

    xnew = pd.DataFrame(data)

    try:
        # Preprocess the input
        xnew_pre = pre.transform(xnew)

        # Match the feature names used when fitting the model
        expected_columns = [
            "con__Air temperature _K_",
            "con__Process temperature _K_",
            "con__Rotational speed _rpm_",
            "con__Torque _Nm_",
            "con__Tool wear _min_"
        ]

        xnew_pre = pd.DataFrame(
            xnew_pre,
            columns=expected_columns
        )

        # Predict
        preds = model.predict(xnew_pre)

        if preds[0] == 1:
            st.error("Potential Machine Failure Detected")
        else:
            st.success("No Machine Failure Predicted")

    except Exception as e:
        st.error(f"Prediction error: {e}")


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set the tab title
st.set_page_config(page_title="Machine Failure Prediction")

# Page title
st.title("Machine Failure Prediction Project")
st.subheader("By Sakshi Chaudhari")

# Load the saved preprocessing pipeline and model
try:
    pre = joblib.load("preprocessing_pipeline.pkl")
    model = joblib.load("gradient_boosting_model.pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# User input fields
Air_Temperature = st.number_input(
    "Air Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=298.0
)

Process_Temperature = st.number_input(
    "Process Temperature [K]",
    min_value=250.0,
    max_value=350.0,
    value=308.0
)

Rotational_Speed = st.number_input(
    "Rotational Speed [rpm]",
    min_value=0,
    value=1500
)

Torque = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    value=40.0
)

Tool_Wear = st.number_input(
    "Tool Wear [min]",
    min_value=0,
    value=100
)

# Prediction button
submit = st.button("Predict Machine Failure")

if submit:

    # Create input DataFrame using original feature names
    data = {
        "Air temperature [K]": [Air_Temperature],
        "Process temperature [K]": [Process_Temperature],
        "Rotational speed [rpm]": [Rotational_Speed],
        "Torque [Nm]": [Torque],
        "Tool wear [min]": [Tool_Wear]
    }

    xnew = pd.DataFrame(data)

    try:
        # Apply the saved preprocessing pipeline
        xnew_pre = pre.transform(xnew)

        # Convert preprocessed output to a DataFrame
        expected_columns = [
            "con__Air temperature _K_",
            "con__Process temperature _K_",
            "con__Rotational speed _rpm_",
            "con__Torque _Nm_",
            "con__Tool wear _min_"
        ]

        if isinstance(xnew_pre, pd.DataFrame):
            xnew_pre = xnew_pre.copy()
            xnew_pre.columns = expected_columns
        else:
            xnew_pre = pd.DataFrame(
                xnew_pre,
                columns=expected_columns
            )

        # Check for missing or infinite values
        if not np.isfinite(xnew_pre.to_numpy(dtype=float)).all():
            st.error(
                "Preprocessing generated missing or infinite values. "
                "Please check the saved preprocessing pipeline."
            )

        else:
            # Predict using the trained model
            preds = model.predict(xnew_pre)

            # Display prediction
            if preds[0] == 1:
                st.error("Potential Machine Failure Detected")
            else:
                st.success("No Machine Failure Predicted")

    except Exception as e:
        st.error(f"Prediction error: {e}")

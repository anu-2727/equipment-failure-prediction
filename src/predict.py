import os
import joblib
import pandas as pd
import numpy as np

# Project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load saved model files
model = joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
threshold = joblib.load(os.path.join(BASE_DIR, "models", "best_threshold.pkl"))
feature_cols = joblib.load(os.path.join(BASE_DIR, "models", "feature_columns.pkl"))


def predict_failure(air_temp, process_temp, rot_speed, torque, tool_wear, product_type="M"):
    """
    Predict equipment failure.
    product_type: 'L', 'M', or 'H'
    """

    # Raw input (must match training column names)
    raw = {
        "Air temperature K": air_temp,
        "Process temperature K": process_temp,
        "Rotational speed rpm": rot_speed,
        "Torque Nm": torque,
        "Tool wear min": tool_wear
    }

    # Feature Engineering
    raw["Temp_Differential"] = process_temp - air_temp
    raw["Power_W"] = torque * (rot_speed * 2 * np.pi / 60)
    raw["Torque_Speed_Ratio"] = torque / (rot_speed + 1)

    # Create DataFrame
    df_input = pd.DataFrame([raw])

    # Product Type Encoding
    df_input["Type_L"] = 1 if product_type == "L" else 0
    df_input["Type_M"] = 1 if product_type == "M" else 0

    # Wear Stage Encoding
    if tool_wear <= 50:
        stage = "Fresh"
    elif tool_wear <= 150:
        stage = "Moderate"
    elif tool_wear <= 200:
        stage = "Worn"
    else:
        stage = "Critical"

    df_input["Wear_Stage_Moderate"] = 1 if stage == "Moderate" else 0
    df_input["Wear_Stage_Worn"] = 1 if stage == "Worn" else 0
    df_input["Wear_Stage_Critical"] = 1 if stage == "Critical" else 0

    # Add missing columns
    for col in feature_cols:
        if col not in df_input.columns:
            df_input[col] = 0

    # Arrange columns in training order
    df_input = df_input[feature_cols]

    # Prediction
    probability = model.predict_proba(df_input)[0][1]
    prediction = int(probability >= threshold)

    return {
        "failure_probability": round(float(probability), 4),
        "prediction": "FAILURE RISK" if prediction else "NORMAL",
        "threshold_used": round(float(threshold), 3)
    }


if __name__ == "__main__":

    result = predict_failure(
        air_temp=302.5,
        process_temp=311.2,
        rot_speed=1450,
        torque=55.3,
        tool_wear=180,
        product_type="M"
    )

    print(result)
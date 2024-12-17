import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os

def generate_global_metadata(filepath, encoder_path, scaler_path):
    """
    Generate and save global preprocessing metadata (encoder and scaler).

    Args:
        filepath (str): Path to the global dataset.
        encoder_path (str): Path to save the OneHotEncoder metadata.
        scaler_path (str): Path to save the MinMaxScaler metadata.
    """
    df = pd.read_csv(filepath)

    # Drop unnecessary columns and handle missing values
    df.drop("customerID", axis=1, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)

    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    encoder = OneHotEncoder(sparse_output=False, drop="first").fit(df[categorical_cols])
    
    # Ensure directory for encoder exists
    os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
    joblib.dump(encoder, encoder_path)

    # Scale numerical variables
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler().fit(df[numerical_cols])

    # Ensure directory for scaler exists
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

if __name__ == "__main__":
    generate_global_metadata(
        filepath="data/Telco-Customer-Churn.csv",
        encoder_path="federated_learning/client/metadata/encoder.pkl",
        scaler_path="federated_learning/client/metadata/scaler.pkl",
    )
    print("Global metadata generated and saved.")
import pandas as pd
import joblib
import os

def preprocess_data(filepath, encoder_path, scaler_path):
    # Leer el archivo CSV
    df = pd.read_csv(filepath)
    print(f"Initial dataset shape: {df.shape}")

    # Eliminar la columna 'customerID' si existe
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
        print("Dropped 'customerID' column.")

    # Convertir 'TotalCharges' a numérico
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    else:
        raise ValueError("Error: 'TotalCharges' column is missing from dataset.")

    # Validar valores nulos iniciales
    initial_nan_count = df.isnull().sum().sum()
    if initial_nan_count > 0:
        print(f"Dataset contains {initial_nan_count} NaN values before processing.")

    # Eliminar filas con valores nulos iniciales
    initial_row_count = df.shape[0]
    df.dropna(inplace=True)
    print(f"Dropped {initial_row_count - df.shape[0]} rows due to missing values.")

    # Verificar si los archivos del encoder y scaler existen
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder file not found at {encoder_path}.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")

    # Cargar encoder y scaler
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    # Manejar valores faltantes en columnas categóricas
    categorical_cols = df.select_dtypes(include=["category", "object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    # Transformar columnas categóricas
    try:
        encoded_data = encoder.transform(df[categorical_cols])
    except ValueError as e:
        print(f"Error during encoding: {e}")
        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")
        encoded_data = encoder.transform(df[categorical_cols])

    # Crear DataFrame codificado y fusionar
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)

    # Renombrar la columna 'Churn_Yes' a 'Churn'
    if "Churn_Yes" in df.columns:
        df.rename(columns={"Churn_Yes": "Churn"}, inplace=True)

    # Manejar valores faltantes en columnas numéricas
    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())

    # Escalar columnas numéricas
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Validar si quedan valores nulos
    if df.isnull().sum().sum() > 0:
        print("Detected NaN values after preprocessing. Dropping problematic rows.")
        df.dropna(inplace=True)

    # Separar características (X) y objetivo (y)
    X = df.drop("Churn", axis=1).values
    y = df["Churn"].values
    print(f"Final dataset shape: Features={X.shape}, Target={y.shape}")
    return X, y
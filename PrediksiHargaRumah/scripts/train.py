import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import joblib
import os
import sys

# Paths
ROOT_DIR = os.getcwd()
DATA_PATH = os.path.join(ROOT_DIR, "data", "raw", "DATA RUMAH.xlsx")
MODEL_PATH = os.path.join(ROOT_DIR, "api", "models", "production_model.pkl")

def train():
    print("Starting training process...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        sys.exit(1)

    # 1. Load Data
    print("Loading data...")
    df = pd.read_excel(DATA_PATH)
    
    # 2. Preprocessing
    # Filter features
    features = ["LB", "LT", "KT", "KM", "GRS"]
    target = "HARGA"
    
    # Simple cleaning: Drop NaNs
    df = df.dropna(subset=features + [target])
    
    # Outlier Removal (Simple IQR) for Price
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]
    print(f"Data shape after cleaning: {df_clean.shape} (Original: {df.shape})")
    
    X = df_clean[features]
    y = df_clean[target]
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train
    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"MAPE: {mape:.2%}")
    print(f"R2 Score: {r2:.4f}")
    
    # 6. Save Model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # 7. Save Metrics
    import json
    import datetime
    
    metrics_path = os.path.join(ROOT_DIR, "api", "models", "metrics.json")
    metrics_data = {
        "mape": mape,
        "r2": r2,
        "last_updated": datetime.datetime.now().strftime("%d %B %Y %H:%M")
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    train()

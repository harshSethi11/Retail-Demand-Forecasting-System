# train_models.py
# Trains a LightGBM model using the processed retail features.

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROCESSED_FILE = "./data/processed/retail_features.parquet"
MODEL_PATH = "./models/lightgbm_sales_model.txt"

os.makedirs("./models", exist_ok=True)


# ----------------------------------------
# Load processed data
# ----------------------------------------
def load_data():
    print("📥 Loading processed features...")
    df = pd.read_parquet(PROCESSED_FILE)
    print(f"Loaded {len(df):,} rows.")
    return df


# ----------------------------------------
# Prepare dataset (features + splits)
# ----------------------------------------
def prepare_dataset(df):
    print("⚙️ Preparing training dataset...")

    df = df.sort_values(["store_id", "item_id", "date"])

    # target
    y = df["sales"].values

    # FEATURES LIST – modify if needed
    feature_cols = [
        "price",
        "promo_flag",
        "sales_lag1",
        "sales_lag7",
        "rolling_mean_7",
    ]

    # Extract feature matrix
    X = df[feature_cols].values

    # time-series split
    split_idx = int(len(df) * 0.85)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train size: {len(X_train):,} rows")
    print(f"Valid size: {len(X_val):,} rows")

    return X_train, y_train, X_val, y_val, feature_cols


# ----------------------------------------
# Train LightGBM model
# ----------------------------------------
def train_lightgbm(X_train, y_train, X_val, y_val, feature_cols):
    print("🚀 Training LightGBM model...")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 32,
        "max_depth": -1,
        "boosting_type": "gbdt",
        "verbose": -1,
        "device": "cpu",  # For compatibility
    }

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols)

    model = lgb.train(
        params,
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=800,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ]
    )

    print(f"💾 Model saved to {MODEL_PATH}")
    model.save_model(MODEL_PATH)

    return model


# ----------------------------------------
# Evaluation
# ----------------------------------------
def evaluate(model, X_val, y_val):
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    smape = 100 * np.mean(
        2 * np.abs(preds - y_val) / (np.abs(y_val) + np.abs(preds) + 1e-8)
    )

    wape = np.sum(np.abs(y_val - preds)) / (np.sum(np.abs(y_val)) + 1e-8)

    print("\n📊 Final Evaluation Metrics")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"SMAPE: {smape:.2f}%")
    print(f"WAPE:  {wape:.4f}")


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == "__main__":
    df = load_data()
    X_train, y_train, X_val, y_val, feature_cols = prepare_dataset(df)
    model = train_lightgbm(X_train, y_train, X_val, y_val, feature_cols)
    evaluate(model, X_val, y_val)

    print("\n✅ Training complete.")

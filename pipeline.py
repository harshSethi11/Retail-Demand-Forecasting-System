# pipeline.py - Retail forecasting pipeline using Polars (Spark replacement)

import polars as pl
import os

RAW_FILE = "./data/pos_raw.csv"
OUTPUT_FILE = "./data/processed/retail_features.parquet"


def load_data(path):
    print("📥 Loading CSV with Polars (lazy mode)...")
    df = (
        pl.scan_csv(path)
        .with_columns([
            # Date parsing for older Polars versions
            pl.col("date").str.to_date(format="%Y-%m-%d"),

            pl.col("sales").cast(pl.Float64),
            pl.col("price").cast(pl.Float64),
            pl.col("promo_flag").cast(pl.Int64),
        ])
    )
    return df


def feature_engineering(df):
    print("⚙️ Performing feature engineering...")

    df = (
        df
        .sort(["store_id", "item_id", "date"])
        .with_columns([
            # Lag features
            pl.col("sales").shift(1).over(["store_id", "item_id"]).alias("sales_lag1"),
            pl.col("sales").shift(7).over(["store_id", "item_id"]).alias("sales_lag7"),

            # Rolling window
            pl.col("sales")
                .rolling_mean(window_size=7)
                .over(["store_id", "item_id"])
                .alias("rolling_mean_7"),
        ])
        .fill_null(0)
    )

    return df


def save(df):
    print(f"📦 Saving to {OUTPUT_FILE}")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    df.collect().write_parquet(OUTPUT_FILE)
    print("✅ Saved successfully.")


if __name__ == "__main__":
    raw_df = load_data(RAW_FILE)
    processed_df = feature_engineering(raw_df)
    save(processed_df)

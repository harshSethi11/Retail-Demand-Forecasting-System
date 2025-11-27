# prepare_pos_data.py
import os
import pandas as pd
import numpy as np

RAW_DIR = "raw"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(RAW_DIR, "train.csv")
ITEMS_CSV = os.path.join(RAW_DIR, "items.csv")

def load_kaggle():
    print("Loading Kaggle files...")

    # Explicit dtypes to avoid mixed-type warnings
    train = pd.read_csv(
        TRAIN_CSV,
        parse_dates=["date"],
        dtype={
            "store_nbr": "int32",
            "item_nbr": "int32",
            "unit_sales": "float32",
            "onpromotion": "string"   # read as string to clean safely
        },
        low_memory=False
    )

    items = pd.read_csv(
        ITEMS_CSV,
        dtype={"item_nbr": "int32", "family": "string"}
    )

    return train, items


def build_pos_raw(train, items, sample_rows=200_000, random_state=42):

    df = train[["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]].copy()

    df.rename(columns={
        "store_nbr": "store_id",
        "item_nbr": "item_id",
        "unit_sales": "sales",
        "onpromotion": "promo_flag"
    }, inplace=True)

    # Clean promo_flag in a modern way (no fillna warning)
    # Convert everything to lowercase string, handle missing, then map to 0/1
    df["promo_flag"] = (
        df["promo_flag"]
        .astype("string")
        .str.lower()
        .fillna("false")
        .replace({"true": 1, "false": 0, "t": 1, "f": 0, "": 0})
        .astype("int8")
    )

    # Clip negative sales (Favorita has returns)
    df["sales"] = df["sales"].clip(lower=0).astype("float32")

    # Merge item families
    if "family" in items.columns:
        fam_map = items.set_index("item_nbr")["family"].to_dict()
        df["family"] = df["item_id"].map(fam_map).fillna("unknown")
        df["price"] = df["family"].astype("category").cat.codes * 0.5 \
                      + np.random.uniform(1.0, 3.0, size=len(df))
        df["price"] = df["price"].astype("float32")
        df.drop(columns=["family"], inplace=True)
    else:
        df["price"] = np.random.uniform(1.0, 5.0, size=len(df)).astype("float32")

    df.sort_values(["date", "store_id", "item_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Downsample to avoid huge load
    if sample_rows is not None and sample_rows < len(df):
        df = df.sample(sample_rows, random_state=random_state)
        df = df.sort_values(["date", "store_id", "item_id"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    train, items = load_kaggle()
    SAMPLE_SIZE = 200_000  # you can reduce to 20_000 if you want faster runs
    pos_df = build_pos_raw(train, items, sample_rows=SAMPLE_SIZE)

    out_path = os.path.join(OUT_DIR, "pos_raw.csv")
    pos_df.to_csv(out_path, index=False)

    print("Wrote:", out_path)
    print("Shape:", pos_df.shape)

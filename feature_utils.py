# feature_utils.py
import numpy as np
import pandas as pd

FEATURE_COLS = ["price", "promo_flag", "sales_lag1", "sales_lag7", "rolling_mean_7"]

def compute_features_from_recent(recent_sales, price=1.0, promo_flag=0):
    """
    recent_sales: list or array-like of past sales in chronological order (oldest ... newest).
                  Must contain at least 7 values to generate sales_lag7 and rolling_mean_7 properly.
    price: float
    promo_flag: int (0/1) — whether today is in promo (for prediction of next day)
    Returns: dict of feature_name -> value matching FEATURE_COLS order.
    """
    arr = np.array(recent_sales, dtype=float)
    if len(arr) == 0:
        raise ValueError("recent_sales must contain at least one value.")
    # last day sales -> sales_lag1
    sales_lag1 = float(arr[-1]) if len(arr) >= 1 else 0.0
    # sales 7 days ago -> sales_lag7 (if not available 0)
    sales_lag7 = float(arr[-7]) if len(arr) >= 7 else 0.0
    # rolling mean of last 7 days (exclude target day — we assume recent_sales are last n days up to now)
    # we take mean of last 7 values of provided series (if fewer, take mean of available)
    window = arr[-7:] if len(arr) >= 7 else arr
    rolling_mean_7 = float(np.mean(window)) if len(window) > 0 else 0.0

    features = {
        "price": float(price),
        "promo_flag": int(promo_flag),
        "sales_lag1": sales_lag1,
        "sales_lag7": sales_lag7,
        "rolling_mean_7": rolling_mean_7,
    }
    return features

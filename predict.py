# predict.py
import argparse
import json
import lightgbm as lgb
import numpy as np
from feature_utils import compute_features_from_recent, FEATURE_COLS
import os

MODEL_PATH = "./models/lightgbm_sales_model.txt"

def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"LightGBM model not found: {path}")
    model = lgb.Booster(model_file=path)
    return model

def predict_from_recent(recent_sales, price=1.0, promo_flag=0):
    features = compute_features_from_recent(recent_sales, price=price, promo_flag=promo_flag)
    X = np.array([[features[c] for c in FEATURE_COLS]], dtype=float)
    model = load_model()
    pred = model.predict(X)[0]
    return pred, features

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recent", help="Comma-separated recent sales (oldest...newest). Example: 5,2,0,3,6,4,2", required=True)
    p.add_argument("--price", type=float, default=1.0)
    p.add_argument("--promo", type=int, default=0, choices=[0,1])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    recent = [float(x.strip()) for x in args.recent.split(",") if x.strip() != ""]
    pred, feats = predict_from_recent(recent, price=args.price, promo_flag=args.promo)
    output = {"prediction": float(pred), "features": feats}
    print(json.dumps(output, indent=2))

import streamlit as st
import lightgbm as lgb
import numpy as np
from feature_utils import compute_features_from_recent, FEATURE_COLS
import os
import pandas as pd

MODEL_PATH = "./models/lightgbm_sales_model.txt"

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}. Run training first.")
        st.stop()
    return lgb.Booster(model_file=path)

def main():
    st.title("Retail Demand Forecast — LightGBM demo")

    st.markdown("Enter recent sales (oldest → newest). For a realistic result, give at least 7 days.")
    recent_str = st.text_area("Recent sales (comma separated)", value="10,12,11,9,13,14,12", height=80)
    price = st.number_input("Price", min_value=0.0, value=2.5, step=0.1)
    promo = st.selectbox("Promo flag", [0,1], index=0)

    if st.button("Predict"):
        try:
            recent = [float(x.strip()) for x in recent_str.split(",") if x.strip() != ""]
            model = load_model()
            feats = compute_features_from_recent(recent, price=price, promo_flag=promo)
            X = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
            pred = float(model.predict(X)[0])
            st.metric("Predicted sales (next day)", f"{pred:.2f}")
            st.write("Features used:")
            st.json(feats)

            # Feature importance 
            fimp = model.feature_importance(importance_type="gain")
            names = model.feature_name()
            if len(names) == len(fimp):
                df_imp = pd.DataFrame({"feature": names, "importance": fimp}).sort_values("importance", ascending=False)
                st.subheader("Feature importance (gain)")
                st.table(df_imp)
            else:
                st.info("Feature importance not available for this model.")
        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()


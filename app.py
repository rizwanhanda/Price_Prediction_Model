import streamlit as st
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb

# --- LOAD THE UPDATED SYSTEM BUNDLE ---
@st.cache_resource
def load_system():
    with open('project_bundle.pkl', 'rb') as f:
        return pickle.load(f)

bundle = load_system()
model_v = bundle['model_validator']
model_i = bundle['model_intel']
feats_v = bundle['features_validator']
feats_i = bundle['features_intel']
cat_options = bundle['category_options']
brand_options = bundle['brand_options']

st.title("🚀 AI Pricing Intelligence Dashboard")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Product Configuration")
engine_mode = st.sidebar.radio("AI Engine:", ["🛡️ Validator (Audit)", "🧠 Intelligence (Strategy)"])

# Shared Inputs
input_cat = st.sidebar.selectbox("Category", cat_options)
input_brand = st.sidebar.selectbox("Brand", brand_options + ["Other"])
input_rating = st.sidebar.slider("Rating", 1.0, 5.0, 4.2)
input_sales = st.sidebar.number_input("Monthly Sales", value=1000)

# Hardware Spec Inputs (The new Intelligence Features)
ram = st.sidebar.number_input("RAM (GB)", 0, 128, 8)
storage = st.sidebar.number_input("Storage (GB)", 0, 2048, 256)
inches = st.sidebar.number_input("Screen Size (Inches)", 0.0, 100.0, 14.0)
is_wireless = st.sidebar.checkbox("Wireless/Bluetooth?")

if engine_mode.startswith("🛡️"):
    input_price = st.sidebar.number_input("Actual Price ($)", value=500.0)
    input_disc = st.sidebar.slider("Discount %", 0, 100, 20)

# --- PREDICTION ---
if st.button("Calculate Market Value"):
    # Create a base dictionary with all columns from the 25-feature Audit mode
    input_dict = {col: 0 for col in feats_v}
    
    # Fill numerical data
    input_dict['sales_volume'] = input_sales
    input_dict['rating'] = input_rating
    input_dict['ram_gb'] = ram
    input_dict['storage_gb'] = storage
    input_dict['screen_inches'] = inches
    input_dict['is_wireless'] = 1 if is_wireless else 0

    # Handle One-Hot Encoding for Category and Brand
    if f"category_{input_cat}" in input_dict: input_dict[f"category_{input_cat}"] = 1
    if f"brand_refined_{input_brand}" in input_dict: input_dict[f"brand_refined_{input_brand}"] = 1

    if engine_mode.startswith("🛡️"):
        input_dict['actual_price'] = input_price
        input_dict['discount_percentage'] = input_disc
        df_final = pd.DataFrame([input_dict])[feats_v]
        prediction = model_v.predict(df_final)[0]
    else:
        # Use the 23-feature Strategy set
        df_final = pd.DataFrame([input_dict])[feats_i]
        log_pred = model_i.predict(df_final)[0]
        # REVERSE THE LOG TRANSFORMATION
        prediction = np.expm1(log_pred)

    st.metric("Predicted Fair Price", f"${prediction:,.2f}")
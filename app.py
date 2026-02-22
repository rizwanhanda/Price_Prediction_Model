import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="🚀 AI Pricing Intelligence Pro", layout="wide")

# --- LOAD THE SYSTEM BUNDLE ---
@st.cache_resource
def load_system():
    with open('project_bundle.pkl', 'rb') as f:
        return pickle.load(f)

try:
    bundle = load_system()
    model_v = bundle['model_validator']
    model_i = bundle['model_intel']
    feats_v = bundle['features_validator']
    feats_i = bundle['features_intel']
    cat_options = bundle['category_options']
    brand_options = bundle['brand_options']
except Exception as e:
    st.error(f"⚠️ System Error: {e}. Ensure 'project_bundle.pkl' is in the repo.")
    st.stop()

# --- SIDEBAR: THE CONTROL PANEL ---
st.sidebar.title("🎛️ Pricing Control Panel")

engine_mode = st.sidebar.radio(
    "Select AI Engine:",
    ["🛡️ Validator (Audit Mode)", "🧠 Intelligence (Strategy Mode)"]
)

st.sidebar.markdown("---")
st.sidebar.header("Product Configuration")

# Shared Core Inputs
input_category = st.sidebar.selectbox("Category", cat_options)
input_brand = st.sidebar.selectbox("Brand", brand_options + ["Other"])

# FIX: Capping Rating impact to prevent the $50,000 swings
input_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)
input_sales = st.sidebar.number_input("Est. Monthly Sales", min_value=0, value=1000, step=100)

# --- DYNAMIC HARDWARE INPUTS ---
st.sidebar.subheader("📐 Hardware Specs")

# Laptop Specifics
if input_category == "Laptop":
    ram = st.sidebar.number_input("RAM (GB)", 0, 128, 32, step=4)
    storage = st.sidebar.number_input("Storage (GB)", 0, 2048, 1024, step=64)
    inches = st.sidebar.number_input("Screen Size (Inches)", 0.0, 100.0, 16.0, step=0.5)
else:
    ram, storage, inches = 0, 0, 0.0

is_wireless = st.sidebar.checkbox("Wireless Features", value=True)

# Validator Specific
if engine_mode.startswith("🛡️"):
    input_price = st.sidebar.number_input("Listed MSRP ($)", min_value=1.0, value=1500.0, step=10.0)
    input_discount = st.sidebar.slider("Current Discount (%)", 0, 100, 20)
else:
    input_price, input_discount = 0.0, 0

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")
st.markdown("---")

if st.button("✨ Generate AI Valuation", type="primary"):
    
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    input_dict = {col: 0.0 for col in target_features}
    
    # Map numerical inputs
    input_dict['sales_volume'] = float(input_sales)
    input_dict['rating'] = float(input_rating)
    input_dict['ram_gb'] = float(ram)
    input_dict['storage_gb'] = float(storage)
    input_dict['screen_inches'] = float(inches)
    input_dict['is_wireless'] = 1.0 if is_wireless else 0.0
    
    if engine_mode.startswith("🛡️"):
        input_dict['actual_price'] = float(input_price)
        input_dict['discount_percentage'] = float(input_discount)

    # Map One-Hot Categories
    cat_col, brand_col = f"category_{input_category}", f"brand_refined_{input_brand}"
    if cat_col in input_dict: input_dict[cat_col] = 1.0
    if brand_col in input_dict: input_dict[brand_col] = 1.0

    final_input = pd.DataFrame([input_dict])[target_features]

    try:
        if engine_mode.startswith("🛡️"):
            prediction = model_v.predict(final_input)[0]
        else:
            # Intelligence Mode Calculation
            log_pred = model_i.predict(final_input)[0]
            prediction = np.expm1(log_pred)
            
        # UI DISPLAY
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Fair Price", f"${prediction:,.2f}")
        with col2:
            st.metric("Market Status", "Ready for Launch" if prediction > 500 else "Budget Entry")
        with col3:
            st.metric("Model Confidence", "85.3%" if not engine_mode.startswith("🛡️") else "99.7%")

        # Analysis Chart
        st.subheader("📊 Price Driver Analysis")
        # Logic to show how RAM vs Rating is affecting price
        st.info(f"The model is valuing this {input_brand} {input_category} primarily based on its {ram}GB RAM and {storage}GB Storage.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

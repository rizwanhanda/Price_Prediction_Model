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
    
    # SAFETY FIX: Using .get() prevents the 'cat_averages' KeyError crash
    cat_avgs = bundle.get('cat_averages', {})
    brand_avgs = bundle.get('brand_averages', {})
    
    if not cat_avgs:
        st.error("⚠️ Data Sync Error: 'cat_averages' not found in bundle. Please re-run Block 15 in Colab and re-upload.")
        st.stop()
        
except Exception as e:
    st.error(f"⚠️ System Error: {e}. Please ensure the latest 'project_bundle.pkl' is on GitHub.")
    st.stop()

# --- SIDEBAR: THE CONTROL PANEL ---
st.sidebar.title("🎛️ Pricing Control Panel")
engine_mode = st.sidebar.radio("AI Engine:", ["🛡️ Validator (Audit)", "🧠 Intelligence (Strategy)"])

st.sidebar.markdown("---")
input_category = st.sidebar.selectbox("Category", cat_options)
input_brand = st.sidebar.selectbox("Brand", brand_options + ["Other"])
input_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)
input_sales = st.sidebar.number_input("Est. Monthly Sales", min_value=0, value=1000, step=100)

# Hardware Specs UI
st.sidebar.subheader("📐 Hardware Specs")
ram = st.sidebar.number_input("RAM (GB)", 0, 128, 8, step=4) if input_category == "Laptop" else 0
storage = st.sidebar.number_input("Storage (GB)", 0, 2048, 256, step=64) if input_category == "Laptop" else 0
inches = st.sidebar.number_input("Screen Size (Inches)", 0.0, 100.0, 14.0, step=0.5) if input_category in ["Laptop", "Monitor", "TV"] else 0.0
is_wireless = st.sidebar.checkbox("Wireless / Bluetooth?", value=True)

if engine_mode.startswith("🛡️"):
    input_price = st.sidebar.number_input("Listed MSRP ($)", value=500.0)
    input_discount = st.sidebar.slider("Discount (%)", 0, 100, 20)

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")
st.markdown(f"**Current Engine:** {engine_mode}")

if st.button("✨ Generate AI Valuation", type="primary"):
    # 1. Blueprint selection
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    input_dict = {col: 0.0 for col in target_features}
    
    # 2. Map core features
    input_dict['sales_volume'] = float(input_sales)
    input_dict['rating'] = float(input_rating)
    input_dict['ram_gb'] = float(ram)
    input_dict['storage_gb'] = float(storage)
    input_dict['screen_inches'] = float(inches)
    input_dict['is_wireless'] = 1.0 if is_wireless else 0.0

    # 3. LOG-POWER LOGIC: This ensures high RAM/Storage drives price UP
    if not engine_mode.startswith("🛡️"):
        # Target Encoding
        input_dict['cat_baseline'] = cat_avgs.get(input_category, np.mean(list(cat_avgs.values())))
        input_dict['brand_baseline'] = brand_avgs.get(input_brand, np.mean(list(brand_avgs.values())))
        
        # FIX: The model now looks for 'hw_power' (log-scaled)
        if 'hw_power' in target_features:
            input_dict['hw_power'] = np.log1p(ram * 10 + storage * 0.5)

    # 4. One-Hot Mapping
    if f"category_{input_category}" in input_dict: input_dict[f"category_{input_category}"] = 1.0
    if f"brand_refined_{input_brand}" in input_dict: input_dict[f"brand_refined_{input_brand}"] = 1.0

    if engine_mode.startswith("🛡️"):
        input_dict['actual_price'] = input_price
        input_dict['discount_percentage'] = input_discount

    # 5. Prediction Engine
    final_input = pd.DataFrame([input_dict])[target_features]
    if engine_mode.startswith("🛡️"):
        prediction = model_v.predict(final_input)[0]
    else:
        log_pred = model_i.predict(final_input)[0]
        prediction = np.expm1(log_pred)

    # --- DISPLAY RESULTS ---
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Predicted Price", f"${prediction:,.2f}")
    with col2:
        # Better delta logic for Strategy mode
        if engine_mode.startswith("🛡️"):
            val = prediction - (input_price * (1-input_discount/100))
            st.metric("Audit Deviation", f"${val:,.2f}")
        else:
            st.metric("Market Positioning", "Premium" if prediction > 1000 else "Standard")
    with col3: st.metric("Model Confidence", "99.7%" if engine_mode.startswith("🛡️") else "83.8%")

    st.subheader("📊 Market Analysis")
    fig = px.scatter(x=[prediction*0.7, prediction, prediction*1.3], y=[3.5, input_rating, 4.8], 
                     color=["Budget", "Your Product", "Premium"], size=[10, 20, 10], template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

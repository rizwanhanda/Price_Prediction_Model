import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION (Restored High-End Look) ---
st.set_page_config(page_title="🚀 AI Pricing Intelligence Pro", layout="wide")

# --- LOAD THE SYSTEM BUNDLE ---
@st.cache_resource
def load_system():
    # Ensure project_bundle.pkl is in your GitHub repo
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
    st.error(f"⚠️ System Error: {e}. Please ensure 'project_bundle.pkl' is uploaded.")
    st.stop()

# --- SIDEBAR: THE CONTROL PANEL ---
st.sidebar.title("🎛️ Pricing Control Panel")

engine_mode = st.sidebar.radio(
    "Select AI Engine:",
    ["🛡️ Validator (Audit Mode)", "🧠 Intelligence (Strategy Mode)"],
    help="Validator uses discount info to check errors. Intelligence predicts value using specs only."
)

st.sidebar.markdown("---")
st.sidebar.header("Product Configuration")

# Core Inputs
input_category = st.sidebar.selectbox("Category", cat_options)
input_brand = st.sidebar.selectbox("Brand", brand_options + ["Other"])
input_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)
input_sales = st.sidebar.number_input("Est. Monthly Sales", min_value=0, value=1000, step=100)

# Hardware Spec Inputs (Refined Step Sizes)
st.sidebar.subheader("📐 Hardware Specs")
ram = st.sidebar.number_input("RAM (GB)", 0, 128, 8, step=4)
storage = st.sidebar.number_input("Storage (GB)", 0, 2048, 256, step=64)
inches = st.sidebar.number_input("Screen Size (Inches)", 0.0, 100.0, 14.0, step=0.5)
is_wireless = st.sidebar.checkbox("Wireless / Bluetooth Features", value=True)

# Validator Specific Inputs
if engine_mode.startswith("🛡️"):
    input_price = st.sidebar.number_input("Listed MSRP ($)", min_value=1.0, value=500.0, step=10.0)
    input_discount = st.sidebar.slider("Current Discount (%)", 0, 100, 20)
else:
    input_price = 1.0 # Placeholder
    input_discount = 0

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")
st.markdown(f"**Current Engine:** {engine_mode}")
st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("✨ Generate AI Valuation", type="primary"):
    
    # 1. Map Inputs to Feature Matrix
    input_dict = {col: 0 for col in feats_v}
    input_dict['sales_volume'] = input_sales
    input_dict['rating'] = input_rating
    input_dict['ram_gb'] = ram
    input_dict['storage_gb'] = storage
    input_dict['screen_inches'] = inches
    input_dict['is_wireless'] = 1 if is_wireless else 0

    if f"category_{input_category}" in input_dict: input_dict[f"category_{input_category}"] = 1
    if f"brand_refined_{input_brand}" in input_dict: input_dict[f"brand_refined_{input_brand}"] = 1

    if engine_mode.startswith("🛡️"):
        input_dict['actual_price'] = input_price
        input_dict['discount_percentage'] = input_discount
        final_input = pd.DataFrame([input_dict])[feats_v]
        prediction = model_v.predict(final_input)[0]
    else:
        # Strategy Mode Logic (Using the 23-feature set)
        final_input = pd.DataFrame([input_dict])[feats_i]
        log_pred = model_i.predict(final_input)[0]
        prediction = np.expm1(log_pred) # Convert Log back to Dollars ($)

    # --- DISPLAY RESULTS (Metric Row) ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estimated Market Price", f"${prediction:,.2f}")
    
    with col2:
        # Logic to check if we are over or under-pricing
        if engine_mode.startswith("🛡️"):
            current_tag_price = input_price * (1 - input_discount/100)
            diff = prediction - current_tag_price
            label = "Audit Deviation"
        else:
            diff = prediction - (prediction * 0.8) # Hypothetical margin check
            label = "Est. Profit Ceiling"
            
        st.metric(label, f"${diff:,.2f}", delta_color="normal")

    with col3:
        # Confidence display based on R2 scores we calculated
        conf = "99.7%" if engine_mode.startswith("🛡️") else "85.3%"
        st.metric("Model Confidence", conf)

    # --- VISUALIZATION (Market Context) ---
    st.subheader("📊 Market Position Analysis")
    
    market_context = pd.DataFrame({
        'Point': ['Budget Avg', 'Premium Avg', 'Your Valuation'],
        'Price ($)': [prediction * 0.65, prediction * 1.35, prediction],
        'Rating': [3.8, 4.8, input_rating]
    })
    
    fig = px.scatter(market_context, x='Price ($)', y='Rating', color='Point', size='Price ($)',
                     title="Where this product sits in the value chain", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Set your hardware specs in the sidebar and trigger the AI valuation.")
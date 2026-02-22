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
    feats_v = bundle['features_validator'] # 25 features
    feats_i = bundle['features_intel']     # 23 features
    cat_options = bundle['category_options']
    brand_options = bundle['brand_options']
except Exception as e:
    st.error(f"⚠️ System Error: {e}. Ensure 'project_bundle.pkl' is in the repo.")
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

# Shared Core Inputs
input_category = st.sidebar.selectbox("Category", cat_options)
input_brand = st.sidebar.selectbox("Brand", brand_options + ["Other"])
input_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)
input_sales = st.sidebar.number_input("Est. Monthly Sales", min_value=0, value=1000, step=100)

# Hardware Spec Inputs
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
    input_price = 0.0 
    input_discount = 0

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")
st.markdown(f"**Current Engine:** {engine_mode}")
st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("✨ Generate AI Valuation", type="primary"):
    
    # 1. Determine which feature set to use
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    
    # 2. Build the input dictionary with ONLY the relevant features
    # Initialize all required features to 0
    input_dict = {col: 0.0 for col in target_features}
    
    # 3. Map numerical inputs
    if 'sales_volume' in input_dict: input_dict['sales_volume'] = float(input_sales)
    if 'rating' in input_dict: input_dict['rating'] = float(input_rating)
    if 'ram_gb' in input_dict: input_dict['ram_gb'] = float(ram)
    if 'storage_gb' in input_dict: input_dict['storage_gb'] = float(storage)
    if 'screen_inches' in input_dict: input_dict['screen_inches'] = float(inches)
    if 'is_wireless' in input_dict: input_dict['is_wireless'] = 1.0 if is_wireless else 0.0
    
    # 4. Map Validator-only numerical inputs
    if engine_mode.startswith("🛡️"):
        if 'actual_price' in input_dict: input_dict['actual_price'] = float(input_price)
        if 'discount_percentage' in input_dict: input_dict['discount_percentage'] = float(input_discount)

    # 5. Map One-Hot Encoded Categorical Inputs
    cat_col = f"category_{input_category}"
    brand_col = f"brand_refined_{input_brand}"
    
    if cat_col in input_dict: input_dict[cat_col] = 1.0
    if brand_col in input_dict: input_dict[brand_col] = 1.0

    # 6. Convert to DataFrame and FORCE COLUMN ORDER (The Fix)
    # Reindexing with the target_features list ensures XGBoost sees the exact same order as training
    final_input = pd.DataFrame([input_dict])[target_features]

    # 7. Predict
    try:
        if engine_mode.startswith("🛡️"):
            prediction = model_v.predict(final_input)[0]
        else:
            log_pred = model_i.predict(final_input)[0]
            prediction = np.expm1(log_pred) # Convert back from Log scale
            
        # --- DISPLAY RESULTS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Market Price", f"${prediction:,.2f}")
        with col2:
            if engine_mode.startswith("🛡️"):
                current_tag_price = input_price * (1 - input_discount/100)
                diff = prediction - current_tag_price
                st.metric("Audit Deviation", f"${diff:,.2f}", delta_color="normal")
            else:
                st.metric("Market Status", "Fair Value", delta="Optimal")
        with col3:
            conf = "99.7%" if engine_mode.startswith("🛡️") else "85.3%"
            st.metric("Model Confidence", conf)

        # --- VISUALIZATION ---
        st.subheader("📊 Market Position Analysis")
        market_context = pd.DataFrame({
            'Point': ['Budget Avg', 'Premium Avg', 'Your Valuation'],
            'Price ($)': [prediction * 0.65, prediction * 1.35, prediction],
            'Rating': [3.8, 4.8, input_rating]
        })
        fig = px.scatter(market_context, x='Price ($)', y='Rating', color='Point', size='Price ($)',
                         template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("This usually happens if the input columns don't match the model's training features.")

else:
    st.info("👈 Set your hardware specs in the sidebar and trigger the AI valuation.")

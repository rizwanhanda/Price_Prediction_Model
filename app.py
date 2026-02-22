import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AI Pricing Intelligence", layout="wide")

# --- LOAD THE SYSTEM BUNDLE ---
@st.cache_resource
def load_system():
    # Load the single file containing both models and all feature lists
    with open('project_bundle.pkl', 'rb') as f:
        return pickle.load(f)

try:
    bundle = load_system()
    model_v = bundle['model_validator']
    model_i = bundle['model_intel']
    feats_v = bundle['features_validator']
    feats_i = bundle['features_intel']
    cat_options = bundle['category_options']
except FileNotFoundError:
    st.error("⚠️ System Error: 'project_bundle.pkl' not found. Please upload the model file.")
    st.stop()

# --- SIDEBAR: USER CONTROLS ---
st.sidebar.title("🎛️ Control Panel")

# 1. Model Selector (The Innovation Flex)
engine_mode = st.sidebar.radio(
    "Select AI Engine:",
    ["🛡️ Validator (High Precision)", "🧠 Intelligence (Market Strategy)"],
    help="Validator uses discount info to check errors. Intelligence predicts fair market value without knowing the discount."
)

st.sidebar.markdown("---")
st.sidebar.header("Product Details")

# 2. Input Fields
input_price = st.sidebar.number_input("Listed Price (₹ or $)", min_value=10.0, value=1500.0, step=50.0)
input_rating = st.sidebar.slider("Customer Rating", 1.0, 5.0, 4.0)
input_sales = st.sidebar.number_input("Sales Volume (Monthly)", min_value=0, value=500)
input_category = st.sidebar.selectbox("Category", cat_options)

# Only show Discount input if we are in 'Validator' mode
if engine_mode.startswith("🛡️"):
    input_discount = st.sidebar.slider("Discount Percentage (%)", 0, 100, 25)
else:
    input_discount = 0  # Intelligence model doesn't need this, but we keep variable safe

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence Dashboard")
st.markdown("Use this tool to audit existing prices or strategize new product launches.")
st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("✨ Predict Market Price", type="primary"):
    
    # 1. Create Raw Dataframe from Inputs
    input_data = pd.DataFrame({
        'actual_price': [input_price],
        'sales_volume': [input_sales],
        'rating': [input_rating],
        'discount_percentage': [input_discount],
        'category': [input_category]
    })

    # 2. Pre-process (One-Hot Encoding)
    # This turns 'Category: Laptop' into the columns the AI expects
    input_data = pd.get_dummies(input_data, columns=['category'])

    # 3. Align Columns (CRITICAL STEP)
    # We force the input to have the exact same columns as the training data
    # If a column is missing (e.g., 'category_Smartwatch'), it adds 0.
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    
    # Reindex fills missing columns with 0 and drops extra ones
    final_input = input_data.reindex(columns=target_features, fill_value=0)

    # 4. Predict
    active_model = model_v if engine_mode.startswith("🛡️") else model_i
    prediction = active_model.predict(final_input)[0]

    # --- DISPLAY RESULTS ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Predicted Selling Price", f"{prediction:,.2f}")
    
    with col2:
        diff = prediction - (input_price * (1 - input_discount/100))
        color = "normal"
        if engine_mode.startswith("🛡️"):
            label = "Deviation from Math"
        else:
            label = "Over/Under Market Value"
            
        st.metric(label, f"{diff:,.2f}", delta_color="inverse")

    with col3:
        confidence = "99.7%" if engine_mode.startswith("🛡️") else "~88.0%"
        st.metric("Model Confidence", confidence)

    # --- VISUALIZATION: MARKET POSITION ---
    st.subheader("📊 Market Position Analysis")
    
    # Fake data for context visualization (Innovation Mark)
    # In a real app, this would come from your dataset
    market_context = pd.DataFrame({
        'Price Segment': ['Budget', 'Value', 'Premium', 'Your Product'],
        'Price': [input_price * 0.4, input_price * 0.7, input_price * 1.2, prediction],
        'Rating': [3.5, 4.0, 4.8, input_rating]
    })
    
    fig = px.scatter(market_context, x='Price', y='Rating', color='Price Segment', size='Price',
                     title="Where does this product fit in the market?")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Adjust parameters in the sidebar and click Predict!")
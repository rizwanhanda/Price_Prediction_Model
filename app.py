import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Pricing Intelligence",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

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
except FileNotFoundError:
    st.error("⚠️ System Error: 'project_bundle.pkl' not found. Please upload the model file.")
    st.stop()

# --- SIDEBAR: USER CONTROLS ---
st.sidebar.title("🎛️ Control Panel")

engine_mode = st.sidebar.radio(
    "Select AI Engine:",
    ["🛡️ Validator (Audit Mode)", "🧠 Intelligence (Strategy Mode)"],
    help="Validator uses discount info to check errors. Intelligence predicts fair market value without knowing the discount."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Product Specs")

input_price = st.sidebar.number_input("Listed Price ($)", min_value=10.0, value=1500.0, step=50.0)
input_rating = st.sidebar.slider("Customer Rating (Stars)", 1.0, 5.0, 4.0, step=0.1)
input_sales = st.sidebar.number_input("Monthly Sales Volume", min_value=0, value=500)
input_category = st.sidebar.selectbox("Product Category", cat_options)

if engine_mode.startswith("🛡️"):
    st.sidebar.subheader("🏷️ Discount Info")
    input_discount = st.sidebar.slider("Discount Percentage (%)", 0, 100, 25)
else:
    input_discount = 0

# --- MAIN DASHBOARD ---
st.title("🚀 AI-Powered Pricing Intelligence")
st.markdown("### Decision Support System for Electronics Pricing")

if st.button("✨ Generate Market Analysis", type="primary"):
    
    # --- 1. DATA PREPARATION ---
    input_data = pd.DataFrame({
        'actual_price': [input_price],
        'sales_volume': [input_sales],
        'rating': [input_rating],
        'discount_percentage': [input_discount],
        'category': [input_category]
    })
    
    # One-Hot Encoding & Column Alignment
    input_encoded = pd.get_dummies(input_data, columns=['category'])
    target_features = feats_v if engine_mode.startswith("🛡️") else feats_i
    final_input = input_encoded.reindex(columns=target_features, fill_value=0)
    
    # --- 2. PREDICTION ---
    active_model = model_v if engine_mode.startswith("🛡️") else model_i
    prediction = active_model.predict(final_input)[0]
    
    # Logic Checks
    user_math_price = input_price * (1 - input_discount/100)
    diff = prediction - user_math_price
    
    # --- 3. RESULTS DISPLAY ---
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("AI Predicted Value", f"${prediction:,.2f}", help="The price the AI thinks this product is worth.")
        
    with col2:
        if engine_mode.startswith("🛡️"):
            label = "Deviation from Expected"
            delta_color = "inverse"
        else:
            label = "Market Value Gap"
            delta_color = "normal" 
        st.metric(label, f"${diff:,.2f}", delta_color=delta_color)

    with col3:
        confidence = "99.75%" if engine_mode.startswith("🛡️") else "99.68%"
        st.metric("Model Accuracy", confidence, help="R² Score on Test Data")

    # Verdict Banner
    if not engine_mode.startswith("🛡️"):
        if diff > 50:
            st.success(f"✅ **Strong Buy:** Undervalued by **${diff:,.2f}**.")
        elif diff < -50:
            st.warning(f"⚠️ **Caution:** Overpriced by **${abs(diff):,.2f}**.")
        else:
            st.info("⚖️ **Fair Price:** Matches market expectations.")
    
    # --- 4. VISUALIZATION TABS (FIXED UI) ---
    st.markdown("### 📊 Deep Dive Analysis")
    tab1, tab2 = st.tabs(["📈 Sensitivity (Star Rating)", "🏆 Competitor Benchmarking"])
    
    with tab1:
        # UX FIX: Combined Title and Description to remove the "Yonky" gap
        st.markdown("**How much is a higher rating worth?**")
        
        # Sensitivity Logic
        scenario_data = []
        for r in range(1, 6):
            temp_input = input_data.copy()
            temp_input['rating'] = r
            temp_enc = pd.get_dummies(temp_input, columns=['category'])
            temp_final = temp_enc.reindex(columns=target_features, fill_value=0)
            pred_price = active_model.predict(temp_final)[0]
            scenario_data.append({'Rating': r, 'Predicted Price': pred_price})
            
        df_scenario = pd.DataFrame(scenario_data)
        
        # Improved Plot
        fig_sens = px.line(df_scenario, x='Rating', y='Predicted Price', markers=True, 
                           labels={'Predicted Price': 'Estimated Value ($)'})
        fig_sens.update_layout(xaxis=dict(tickmode='linear', tick0=1, dtick=1)) # Force integer ticks
        
        # Highlight User's Position
        fig_sens.add_trace(go.Scatter(x=[input_rating], y=[prediction], mode='markers', 
                                      marker=dict(size=15, color='red'), name='Current Selection'))
        
        st.plotly_chart(fig_sens, use_container_width=True)

    with tab2:
        st.markdown("**Where does this product fit in the market hierarchy?**")
        
        # Context Data
        np.random.seed(42)
        base_price = input_price
        context_data = pd.DataFrame({
            'Product': ['Competitor A', 'Competitor B', 'Competitor C', 'Market Leader', 'Budget Option', 'YOUR PRODUCT'],
            'Price': [base_price*0.9, base_price*1.1, base_price*0.85, base_price*1.3, base_price*0.6, prediction],
            'Rating': [3.8, 4.2, 3.5, 4.8, 2.9, input_rating],
            'Type': ['Rival', 'Rival', 'Rival', 'Premium', 'Budget', 'Target']
        })
        
        fig_market = px.scatter(context_data, x='Price', y='Rating', color='Type', size='Price',
                                hover_name='Product', size_max=40,
                                color_discrete_map={'Target': 'red', 'Rival': 'blue', 'Premium': 'purple', 'Budget': 'green'})
        fig_market.add_vline(x=prediction, line_dash="dash", line_color="green", annotation_text="AI Fair Value")
        st.plotly_chart(fig_market, use_container_width=True)

    # --- 5. LOGICAL IMPROVEMENT: REPORT DOWNLOAD ---
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Analysis Report",
        data=csv,
        file_name='pricing_analysis.csv',
        mime='text/csv',
    )

else:
    st.info("👈 Please adjust the product details in the sidebar and click **Generate Analysis**.")
    st.caption("Powered by XGBoost | Thapar Institute of Engineering & Technology")

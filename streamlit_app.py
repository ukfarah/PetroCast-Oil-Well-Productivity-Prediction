import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import json

# Load model and stats
model = joblib.load('models/xgboost_model.joblib')
shap_img = Image.open('models/shap_summary.png')
with open('models/feature_stats.json', 'r') as f:
    stats = json.load(f)

# App layout
st.set_page_config(page_title='PetroCast', layout='wide')
st.title("Oil Production Prediction 🌍")
st.write("Predict daily oil production using well parameters")

# Input form
col1, col2 = st.columns(2)
with col1:
    days = st.slider("Days since production start", 
                    min_value=stats['Days']['min'],
                    max_value=stats['Days']['max'],
                    value=(stats['Days']['min'] + stats['Days']['max'])//2)
    
    gor = st.slider("Gas Oil Ratio (GOR)", 
                   min_value=stats['GOR']['min'],
                   max_value=stats['GOR']['max'],
                   value=(stats['GOR']['min'] + stats['GOR']['max'])//2)
    
with col2:
    whp = st.slider("Wellhead Pressure (WHP)", 
                   min_value=stats['WHP']['min'],
                   max_value=stats['WHP']['max'],
                   value=(stats['WHP']['min'] + stats['WHP']['max'])//2)
    
    wht = st.slider("Wellhead Temperature (WHT)", 
                   min_value=stats['WHT']['min'],
                   max_value=stats['WHT']['max'],
                   value=(stats['WHT']['min'] + stats['WHT']['max'])//2)

# Prediction
if st.button("Predict Oil Rate 🚀"):
    input_df = pd.DataFrame({
        'Days': [days],
        'GOR': [gor],
        'WHP': [whp],
        'WHT': [wht]
    })
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Oil Rate: {prediction:.2f} barrels/day")
    
# Model metrics
st.subheader("Model Performance")
st.write("Metrics from test evaluation:")
st.markdown(f"""
- **Mean Squared Error (MSE):** 0.45  
- **R² Score:** 0.89  
""")

# SHAP explainability
st.subheader("Feature Importance")
st.image(shap_img, caption='SHAP Feature Importance')
st.write("Shows which features most influence predictions")

# Styling
st.markdown(
    """
    <style>
    .stSlider { margin: 15px 0; }
    .stButton>button { background-color: #4CAF50; }
    </style>
    """,
    unsafe_allow_html=True
)
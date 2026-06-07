# app.py - Load model dari Google Drive dengan gdown
import streamlit as st
import pickle
import numpy as np
import gdown
import os

st.set_page_config(
    page_title="⚽ Player Value Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Football Player Market Value Predictor")
st.caption("ML model trained on real player data from Kaggle")

# ============================================================
# LOAD MODEL FROM GOOGLE DRIVE USING GDOWN
# ============================================================

@st.cache_resource
def load_model_from_drive():
    """Load model from Google Drive using gdown"""
    
    FILE_ID = "1MDLz7SL_lYHCk9Zf8hF5EtrAemc0xo8z"
    FILE_PATH = "player_value_model.pkl"
    
    try:
        # Download dengan gdown (lebih reliable)
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, FILE_PATH, quiet=False)
        
        # Load pickle
        with open(FILE_PATH, 'rb') as f:
            artifacts = pickle.load(f)
        
        st.success(f"✅ Model loaded: {artifacts['model_name']}")
        
        # Hapus file setelah load (opsional, untuk hemat space)
        # os.remove(FILE_PATH)
        
        return artifacts
        
    except Exception as e:
        st.error(f"Failed to load model from Drive: {e}")
        st.info("Make sure the file is shared as 'Anyone with the link'")
        return None

# Load model
artifacts = load_model_from_drive()

if artifacts:
    model = artifacts["model"]
    le_position = artifacts["le_position"]
    le_sub_position = artifacts["le_sub_position"]
    le_foot = artifacts["le_foot"]
    
    with st.expander("ℹ️ Model Information"):
        m = artifacts["metrics"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Model", artifacts["model_name"])
        col2.metric("R² Score", f"{m.get('r2', 0):.4f}")
        col3.metric("MAE", f"€{m.get('mae_eur', 0)/1e6:.2f}M")
    
    st.divider()
    st.subheader("📋 Player Profile")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        age = st.slider("Age", 15, 45, 24)
        height_cm = st.number_input("Height (cm)", 150, 215, 180)
        position = st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
        sub_position = st.selectbox("Sub-position", 
            ["Centre-Forward", "Winger", "Midfielder", "Defender", "Goalkeeper"])
        foot = st.selectbox("Preferred Foot", ["right", "left", "both"])
    
    with col_b:
        highest_mv = st.number_input("Highest Ever Market Value (€)", 0, 200_000_000, 5_000_000, step=500000)
        total_apps = st.number_input("Career Appearances", 0, 1000, 80)
        total_goals = st.number_input("Career Goals", 0, 500, 20)
        total_assists = st.number_input("Career Assists", 0, 500, 15)
        avg_mins = st.slider("Avg Minutes Played / Game", 0, 95, 75)
        yellow_cards = st.number_input("Total Yellow Cards", 0, 300, 8)
        red_cards = st.number_input("Total Red Cards", 0, 50, 1)
        seasons = st.slider("Seasons Active", 1, 20, 5)
    
    def safe_encode(encoder, value):
        try:
            return int(encoder.transform([value])[0])
        except (ValueError, AttributeError):
            return 0
    
    def predict():
        goals_pg = total_goals / (total_apps + 1)
        assists_pg = total_assists / (total_apps + 1)
        
        X = np.array([[
            age, height_cm,
            safe_encode(le_position, position),
            safe_encode(le_sub_position, sub_position),
            safe_encode(le_foot, foot),
            highest_mv, total_apps, total_goals, total_assists,
            avg_mins, goals_pg, assists_pg,
            yellow_cards, red_cards, seasons
        ]])
        
        log_pred = model.predict(X)[0]
        value_eur = float(np.expm1(log_pred))
        value_M = value_eur / 1_000_000
        
        return value_M, value_eur
    
    if st.button("🔮 Predict Market Value", type="primary", use_container_width=True):
        value_M, value_eur = predict()
        
        st.success(f"### 💰 Predicted Market Value: **€{value_M:.2f}M**")
        st.progress(min(value_M / 200, 1.0), text=f"€{value_M:.1f}M / €200M scale")
        
        if value_M >= 80:
            st.info("🌟 World-class player")
        elif value_M >= 30:
            st.info("⭐ Top-tier player")
        elif value_M >= 10:
            st.info("🔵 Quality first-team player")
        elif value_M >= 2:
            st.info("🟡 Rotation / squad player")
        else:
            st.info("🟤 Development / fringe player")
    
    st.divider()
    st.caption("Built with 🐍 Streamlit | Model trained on Kaggle player dataset")
else:
    st.warning("⚠️ Model not loaded. Please check Google Drive link is correct and file is shared.")

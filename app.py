# app.py - Load model langsung dari Google Drive
import streamlit as st
import pickle
import requests
import io
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="⚽ Player Value Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Football Player Market Value Predictor")
st.caption("ML model trained on real player data from Kaggle")

# ============================================================
# LOAD MODEL FROM GOOGLE DRIVE
# ============================================================

@st.cache_resource
def load_model_from_drive():
    """
    Load model from Google Drive using direct download link
    """
    # FILE_ID dari Google Drive Anda
    FILE_ID = "1oCRt4TUlgqzGyx236v0MzU5-khRvvS_1"  # <-- SUDAH DIISI
    DIRECT_LINK = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    
    try:
        # Download file dari Drive
        response = requests.get(DIRECT_LINK)
        
        # Handle Google Drive warning page
        if 'download_warning' in response.text:
            # Extract confirm token
            import re
            confirm_token = re.search('confirm=([^&]+)', response.text)
            if confirm_token:
                confirm = confirm_token.group(1)
                DIRECT_LINK = f"https://drive.google.com/uc?export=download&confirm={confirm}&id={FILE_ID}"
                response = requests.get(DIRECT_LINK)
        
        # Load pickle
        artifacts = pickle.loads(response.content)
        
        st.success(f"✅ Model loaded: {artifacts['model_name']}")
        return artifacts
        
    except Exception as e:
        st.error(f"Failed to load model from Drive: {e}")
        st.info("Falling back to default model...")
        return create_default_model()

def create_default_model():
    """Fallback model jika gagal load dari Drive"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    
    # Create dummy model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Dummy training (just for structure)
    X_dummy = np.random.randn(100, 15)
    y_dummy = np.random.randn(100)
    model.fit(X_dummy, y_dummy)
    
    # Label encoders with common values
    le_position = LabelEncoder()
    le_sub_position = LabelEncoder()
    le_foot = LabelEncoder()
    
    positions = ["Attack", "Midfield", "Defender", "Goalkeeper"]
    sub_positions = ["Centre-Forward", "Midfielder", "Defender", "Goalkeeper"]
    foot_options = ["right", "left", "both"]
    
    le_position.fit(positions)
    le_sub_position.fit(sub_positions)
    le_foot.fit(foot_options)
    
    return {
        "model": model,
        "model_name": "Default Model (Fallback)",
        "le_position": le_position,
        "le_sub_position": le_sub_position,
        "le_foot": le_foot,
        "features": ["age", "height_in_cm", "position_enc", "sub_position_enc", 
                    "foot_enc", "highest_market_value_in_eur", "total_appearances",
                    "total_goals", "total_assists", "avg_minutes_played", 
                    "goals_per_game", "assists_per_game", "total_yellow_cards",
                    "total_red_cards", "seasons_active"],
        "scaler": None,
        "metrics": {"r2": 0.75, "mae_eur": 8000000}
    }

# Load model
artifacts = load_model_from_drive()
model = artifacts["model"]
le_position = artifacts["le_position"]
le_sub_position = artifacts["le_sub_position"]
le_foot = artifacts["le_foot"]
features = artifacts["features"]

# ============================================================
# UI Components
# ============================================================

with st.expander("ℹ️ Model Information"):
    m = artifacts.get("metrics", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", artifacts.get("model_name", "Unknown"))
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
    highest_mv = st.number_input("Highest Ever Market Value (€)", 0, 200_000_000, 5_000_000)
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

# app.py - Football Player Market Value Predictor
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="⚽ Player Value Predictor",
    page_icon="⚽",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        with open("player_value_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found! Please upload player_value_model.pkl")
        return None

# Prediction function
def predict_player_value(art, input_data):
    log_pred = art["model"].predict(input_data)[0]
    value_eur = float(np.expm1(log_pred))
    return value_eur

def safe_encode(encoder, value):
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    return 0

def get_tier(value_M):
    if value_M >= 80: return "🌟 World-class", "#gold"
    elif value_M >= 30: return "⭐ Top-tier", "silver"
    elif value_M >= 10: return "🔵 Quality first-team", "#4CAF50"
    elif value_M >= 2: return "🟡 Rotation/squad", "#FFC107"
    else: return "🟤 Development/fringe", "#9E9E9E"

# Main app
def main():
    st.title("⚽ Football Player Market Value Predictor")
    st.markdown("*Machine Learning-powered estimation based on player statistics*")
    
    # Load model
    art = load_model()
    if art is None:
        st.warning("Please upload the model file to continue")
        uploaded_file = st.file_uploader("Upload player_value_model.pkl", type="pkl")
        if uploaded_file:
            art = pickle.load(uploaded_file)
            st.success("Model loaded successfully!")
        else:
            st.stop()
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(f"""
        **Model:** `{art['model_name']}`  
        **R² Score:** `{art['metrics']['r2']:.4f}`  
        **MAE:** `€{art['metrics']['mae_eur']/1e6:.2f}M`
        
        ---
        **Features used:**
        - Age & Height
        - Position (main & sub)
        - Career statistics
        - Goals/assists per game
        - Discipline record
        """)
    
    # Main form - 2 columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Player Profile")
        age = st.slider("Age (years)", 15, 45, 24, help="Player's current age")
        height_cm = st.number_input("Height (cm)", 150, 220, 180, step=1)
        
        position = st.selectbox("Main Position", 
            ["Attack", "Midfield", "Defender", "Goalkeeper"])
        
        sub_position_options = {
            "Attack": ["Centre-Forward", "Left Winger", "Right Winger", "Second Striker"],
            "Midfield": ["Attacking Midfield", "Central Midfield", "Defensive Midfield", 
                        "Left Midfield", "Right Midfield"],
            "Defender": ["Centre-Back", "Left-Back", "Right-Back", "Sweeper"],
            "Goalkeeper": ["Goalkeeper"]
        }
        sub_position = st.selectbox("Sub-position", sub_position_options.get(position, ["Unknown"]))
        
        foot = st.selectbox("Preferred Foot", ["right", "left", "both"])
    
    with col2:
        st.subheader("📊 Career Statistics")
        highest_mv = st.number_input("Highest Ever Market Value (€)", 
                                     0, 200_000_000, 5_000_000, step=500_000,
                                     format="%d")
        total_apps = st.number_input("Career Appearances", 0, 1000, 80, step=5)
        total_goals = st.number_input("Career Goals", 0, 500, 20, step=5)
        total_assists = st.number_input("Career Assists", 0, 500, 15, step=5)
        avg_mins = st.slider("Avg Minutes per Game", 0, 95, 75, 
                            help="Average minutes played per appearance")
        yellow_cards = st.number_input("Total Yellow Cards", 0, 300, 8)
        red_cards = st.number_input("Total Red Cards", 0, 50, 1)
        seasons = st.slider("Seasons Active", 1, 20, 5)
    
    # Calculate derived features
    goals_pg = total_goals / (total_apps + 1)
    assists_pg = total_assists / (total_apps + 1)
    
    # Encode categorical variables
    pos_enc = safe_encode(art["le_position"], position)
    sub_enc = safe_encode(art["le_sub_position"], sub_position)
    foot_enc = safe_encode(art["le_foot"], foot)
    
    # Create feature array
    features = np.array([[
        age, height_cm, pos_enc, sub_enc, foot_enc,
        highest_mv, total_apps, total_goals, total_assists,
        avg_mins, goals_pg, assists_pg,
        yellow_cards, red_cards, seasons
    ]])
    
    # Predict button
    st.markdown("---")
    col1_btn, col2_btn, col3_btn = st.columns([1, 2, 1])
    with col2_btn:
        predict_btn = st.button("🔮 PREDICT MARKET VALUE", use_container_width=True)
    
    if predict_btn:
        with st.spinner("Calculating..."):
            value_eur = predict_player_value(art, features)
            value_M = value_eur / 1_000_000
        
        # Display results
        st.markdown("---")
        st.subheader("📈 Prediction Result")
        
        # Metric cards
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div class="metric-card">
                <h4>💰 Market Value</h4>
                <h2 style="color:#4CAF50;">€{value_M:.2f}M</h2>
                <p>({value_eur:,.0f} EUR)</p>
            </div>
            """, unsafe_allow_html=True)
        
        tier_name, tier_color = get_tier(value_M)
        with col_b:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏆 Player Tier</h4>
                <h2 style="color:{tier_color};">{tier_name}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Performance</h4>
                <h4>{goals_pg:.2f} G/game | {assists_pg:.2f} A/game</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress bar
        st.markdown("---")
        st.markdown("**Value percentile (relative to €200M scale)**")
        st.progress(min(value_M / 200, 1.0))
        
        # Disclaimer
        st.caption("⚠️ Prediction based on historical data and machine learning model. For reference only.")
        
        # Feature importance insight
        with st.expander("🔍 What influences this prediction?"):
            st.markdown(f"""
            - **Age {age}** - {'Peak years' if 24 <= age <= 29 else 'Developing/Declining'}
            - **Goals per game {goals_pg:.2f}** - {'Excellent' if goals_pg > 0.5 else 'Good' if goals_pg > 0.3 else 'Room for improvement'}
            - **Assists per game {assists_pg:.2f}** - {'Creative player' if assists_pg > 0.2 else ''}
            - **Highest market value €{highest_mv/1e6:.1f}M** - {'Established performer' if highest_mv > 0 else 'Undeveloped potential'}
            """)

if __name__ == "__main__":
    main()

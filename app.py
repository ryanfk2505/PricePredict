import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="⚽ Player Value Predictor",
    page_icon="⚽",
    layout="centered",
)

st.title("⚽ Football Player Market Value Predictor")
st.caption("ML-powered estimation based on player stats & attributes")

# ── Load model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("player_value_model.pkl", "rb") as f:
        return pickle.load(f)

art = load_model()

# ── Show model info ───────────────────────────────────────
with st.expander("ℹ️ Model Information"):
    m = art["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Model",    art["model_name"])
    col2.metric("R² Score", f"{m['r2']:.4f}")
    col3.metric("MAE",      f"€{m['mae_eur']/1e6:.2f}M")

st.divider()

# ── Input form ────────────────────────────────────────────
st.subheader("📋 Player Profile")

col_a, col_b = st.columns(2)

with col_a:
    age            = st.slider("Age", 15, 45, 24)
    height_cm      = st.number_input("Height (cm)", 150, 215, 180)
    position       = st.selectbox("Position",
        ["Attack", "Midfield", "Defender", "Goalkeeper"])
    sub_position   = st.selectbox("Sub-position",
        ["Centre-Forward", "Left Winger", "Right Winger",
         "Attacking Midfield", "Central Midfield", "Defensive Midfield",
         "Left Midfield", "Right Midfield",
         "Centre-Back", "Left-Back", "Right-Back",
         "Goalkeeper"])
    foot           = st.selectbox("Preferred Foot", ["right", "left", "both"])

with col_b:
    highest_mv     = st.number_input("Highest Ever Market Value (€)",
                                      0, 200_000_000, 5_000_000, step=500_000)
    total_apps     = st.number_input("Career Appearances",    0, 1000, 80)
    total_goals    = st.number_input("Career Goals",          0, 500,  20)
    total_assists  = st.number_input("Career Assists",        0, 500,  15)
    avg_mins       = st.slider("Avg Minutes Played / Game",   0, 95, 75)
    yellow_cards   = st.number_input("Total Yellow Cards",    0, 300, 8)
    red_cards      = st.number_input("Total Red Cards",       0, 50,  1)
    seasons        = st.slider("Seasons Active",              1, 20,  5)

# ── Predict button ────────────────────────────────────────
if st.button("🔮 Predict Market Value", type="primary", use_container_width=True):
    le_pos = art["le_position"]
    le_sub = art["le_sub_position"]
    le_ft  = art["le_foot"]

    def safe_enc(le, val):
        return int(le.transform([val])[0]) if val in le.classes_ else 0

    goals_pg   = total_goals   / (total_apps + 1)
    assists_pg = total_assists / (total_apps + 1)

    X = np.array([[age, height_cm,
                   safe_enc(le_pos, position),
                   safe_enc(le_sub, sub_position),
                   safe_enc(le_ft, foot),
                   highest_mv, total_apps, total_goals, total_assists,
                   avg_mins, goals_pg, assists_pg,
                   yellow_cards, red_cards, seasons]])

    if art["is_linear"]:
        pass

    log_pred  = art["model"].predict(X)[0]
    value_eur = float(np.expm1(log_pred))
    value_M   = value_eur / 1_000_000

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

    st.caption("*Prediction based on classical ML regression model. "
               "For reference purposes only.*")
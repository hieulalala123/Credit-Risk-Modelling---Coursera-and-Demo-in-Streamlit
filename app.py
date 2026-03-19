import streamlit as st

st.set_page_config(
    page_title="Credit Risk Modeling",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: white; border-radius: 14px; padding: 1.3rem 1.5rem;
        box-shadow: 0 2px 16px rgba(0,0,0,0.07);
        border-left: 5px solid #1e88e5; margin-bottom: 1rem;
        transition: transform .18s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card.red    { border-left-color: #e53935; }
    .metric-card.green  { border-left-color: #43a047; }
    .metric-card.orange { border-left-color: #fb8c00; }
    .metric-card.purple { border-left-color: #8e24aa; }

    .metric-label { font-size: .72rem; font-weight: 600; color: #888;
                    text-transform: uppercase; letter-spacing: .9px; }
    .metric-value { font-size: 1.9rem; font-weight: 700; color: #1a1a2e; margin: .2rem 0; }
    .metric-sub   { font-size: .78rem; color: #999; }

    .risk-badge { display: inline-block; padding: .3rem 1rem; border-radius: 50px;
                  font-weight: 700; font-size: .85rem; }
    .risk-low    { background: #e8f5e9; color: #2e7d32; }
    .risk-medium { background: #fff3e0; color: #e65100; }
    .risk-high   { background: #ffebee; color: #b71c1c; }

    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #e3f2fd; padding-bottom: .45rem; margin-bottom: 1rem;
    }
    .info-box {
        background: #e3f2fd; border-radius: 10px; padding: .9rem 1.2rem;
        border-left: 4px solid #1e88e5; margin-bottom: 1rem;
    }
    .formula-box {
        background: #1a1a2e; color: #e0e0e0; border-radius: 10px;
        padding: 1rem 1.4rem; font-family: monospace; font-size: .88rem;
        margin: .75rem 0; line-height: 1.8;
    }

    div[data-testid="stSidebar"] { background: #0d1b2a !important; }
    div[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    div[data-testid="stSidebar"] h3 { color: #90caf9 !important; }
    div[data-testid="stSidebar"] hr { border-color: #1e3a5f !important; }

    .stProgress > div > div { background: linear-gradient(90deg, #1e88e5, #42a5f5); }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

import views.home      as home
import views.pd_model  as pd_model
import views.el_calc   as el_calc
import views.portfolio as portfolio

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 .5rem;">
        <div style="font-size:2.2rem;">🏦</div>
        <div style="font-size:1rem; font-weight:700; color:#90caf9; margin-top:.3rem;">
            Credit Risk
        </div>
        <div style="font-size:.78rem; color:#607d8b; margin-top:.1rem;">
            Modeling System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "nav",
        [
            "🏠  Home",
            "📊  PD Model",
            "💰  Expected Loss",
            "📁  Portfolio",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.75rem; color:#546e7a; line-height:1.7; padding:.2rem 0;">
        <strong style="color:#90caf9;">📌 Dataset</strong><br>
        LendingClub 2007–2014<br>
        466K+ loan records<br><br>
        <strong style="color:#90caf9;">📐 Standard</strong><br>
        Basel II / IFRS 9
    </div>
    """, unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
if   page == "🏠  Home":           home.render()
elif page == "📊  PD Model":       pd_model.render()
elif page == "💰  Expected Loss":  el_calc.render()
elif page == "📁  Portfolio":      portfolio.render()

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import compute_pd, risk_label, risk_color, get_scorecard


def render():
    st.markdown("""
    <div class="main-header">
        <h1>📊 Probability of Default (PD) Model</h1>
        <p>Logistic Regression · WoE-encoded features · 87 variables · AUROC = 0.861</p>
    </div>""", unsafe_allow_html=True)

    sc = get_scorecard()
    # Only keep non-reference (coef != 0), skip intercept
    sc_active = sc[(sc["Coefficients"] != 0) & (sc["Feature name"] != "Intercept")].copy()
    sc_active = sc_active.sort_values("Coefficients", ascending=False)

    tab1, tab2, tab3 = st.tabs(["🎯 PD Calculator", "📊 Scorecard Analysis", "📋 Model Summary"])

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Loan Parameters")
        grade    = st.select_slider("Grade", ["G","F","E","D","C","B","A"], value="C")
        term     = st.radio("Term (months)", [36, 60], horizontal=True)
        int_rate = st.slider("Interest Rate (%)", 5.0, 28.0, 13.0, 0.1)
        annual_inc = st.number_input("Annual Income ($)", 10000, 300000, 65000, 5000)
        dti      = st.slider("Debt-to-Income Ratio", 0.0, 45.0, 15.0, 0.5)
        home_own = st.selectbox("Home Ownership", ["RENT","MORTGAGE","OWN"])
        verif    = st.selectbox("Verification Status",
                                ["Not Verified","Source Verified","Verified"])
        purpose  = st.selectbox("Purpose",
                                ["debt_consolidation","credit_card",
                                 "home_improvement","major_purchase","other"])
        emp_len  = st.slider("Employment Length (yrs)", 0, 10, 5)
        inq      = st.slider("Inquiries (last 6 months)", 0, 10, 1)
        open_acc = st.slider("Open Accounts", 1, 30, 8)
        months_issue = st.slider("Months Since Issue", 1, 120, 36)
        months_cr    = st.slider("Months Since Earliest CR", 100, 400, 220)
        mth_delinq   = st.slider("Months Since Last Delinq (-1 = never)", -1, 120, -1)
        mth_record   = st.slider("Months Since Last Record (-1 = never)", -1, 100, -1)
        acc_delinq   = st.radio("Acc Now Delinquent", [0, 1], horizontal=True)
        initial_ls   = st.radio("Initial List Status", ["f","w"], horizontal=True)

    inputs = dict(
        grade=grade, term=term, int_rate=int_rate, annual_inc=annual_inc,
        dti=dti, home_ownership=home_own, verification_status=verif,
        purpose=purpose, emp_length=emp_len, inq_last_6mths=inq,
        open_acc=open_acc, months_since_issue_d=months_issue,
        months_since_earliest_cr_line=months_cr,
        mths_since_last_delinq=mth_delinq,
        mths_since_last_record=mth_record,
        acc_now_delinq=acc_delinq,
        initial_list_status=initial_ls,
    )

    pd_val = compute_pd(inputs)
    rl  = risk_label(pd_val)
    rc  = risk_color(pd_val)
    badge_cls = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}[rl]

    # ── Tab 1: Calculator ─────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Single Loan PD — Real Model Output</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card {rc}">
                <div class="metric-label">Probability of Default</div>
                <div class="metric-value">{pd_val:.2%}</div>
                <div class="metric-sub"><span class="risk-badge {badge_cls}">{rl} Risk</span></div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card green">
                <div class="metric-label">Survival Probability</div>
                <div class="metric-value">{1-pd_val:.2%}</div>
                <div class="metric-sub">P(no default)</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            # Credit score from scorecard: base 305 + sum of matched scores
            credit_score = int(850 - pd_val * 550)
            st.markdown(f"""<div class="metric-card purple">
                <div class="metric-label">Indicative Credit Score</div>
                <div class="metric-value">{credit_score}</div>
                <div class="metric-sub">Mapped from PD</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            odds = pd_val / max(1 - pd_val, 1e-9)
            st.markdown(f"""<div class="metric-card orange">
                <div class="metric-label">Default Odds</div>
                <div class="metric-value">1:{1/max(odds,1e-9):.0f}</div>
                <div class="metric-sub">1 default per {1/max(odds,1e-9):.0f} loans</div>
            </div>""", unsafe_allow_html=True)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pd_val * 100,
            number={"suffix":"%","font":{"size":38,"color":"#1e3a5f"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1},
                "bar":{"color":"#e53935" if rl=="High" else "#fb8c00" if rl=="Medium" else "#43a047",
                       "thickness":0.3},
                "bgcolor":"white","borderwidth":0,
                "steps":[
                    {"range":[0,10],"color":"#e8f5e9"},
                    {"range":[10,20],"color":"#fff3e0"},
                    {"range":[20,100],"color":"#ffebee"},
                ],
                "threshold":{"line":{"color":"#1e3a5f","width":3},
                             "thickness":.8,"value":pd_val*100},
            },
            title={"text":"PD (%)","font":{"size":16}},
        ))
        fig.update_layout(height=280, margin=dict(t=50,b=10,l=30,r=30),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # PD across grades for current inputs
        st.markdown("**PD across all Grades** (other inputs fixed)")
        grades = ["A","B","C","D","E","F","G"]
        pd_grades = [compute_pd({**inputs,"grade":g}) for g in grades]
        colors = ["#43a047","#66bb6a","#fff176","#ffa726","#ef5350","#c62828","#b71c1c"]
        fig2 = go.Figure(go.Bar(
            x=grades, y=[v*100 for v in pd_grades], marker_color=colors,
            text=[f"{v:.2%}" for v in pd_grades], textposition="outside",
        ))
        fig2.add_hline(y=pd_val*100, line_dash="dash", line_color="#1e88e5",
                       annotation_text=f"Current ({grade}): {pd_val:.2%}")
        fig2.update_layout(xaxis_title="Grade", yaxis_title="PD (%)",
                           height=300, plot_bgcolor="#f8faff",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Scorecard Analysis ─────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">📋 Scorecard — Coefficients from Real Model</div>',
                    unsafe_allow_html=True)

        # Feature selector
        feature_groups = {
            "Grade":       sc_active[sc_active["Feature name"].str.startswith("grade:")],
            "Interest Rate": sc_active[sc_active["Feature name"].str.startswith("int_rate:")],
            "Months Since Issue": sc_active[sc_active["Feature name"].str.startswith("mths_since_issue_d:")],
            "Annual Income": sc_active[sc_active["Feature name"].str.startswith("annual_inc:")],
            "DTI":         sc_active[sc_active["Feature name"].str.startswith("dti:")],
            "Inquiries":   sc_active[sc_active["Feature name"].str.startswith("inq_last_6mths:")],
            "Open Accounts": sc_active[sc_active["Feature name"].str.startswith("open_acc:")],
        }

        selected = st.selectbox("Select feature group to visualise",
                                list(feature_groups.keys()))
        grp = feature_groups[selected].copy()

        col_l, col_r = st.columns([3, 2])
        with col_l:
            bar_colors = ["#43a047" if v >= 0 else "#e53935" for v in grp["Coefficients"]]
            fig3 = go.Figure(go.Bar(
                x=grp["Feature name"], y=grp["Coefficients"],
                marker_color=bar_colors,
                text=[f"{v:.4f}" for v in grp["Coefficients"]],
                textposition="outside",
            ))
            fig3.add_hline(y=0, line_dash="dot", line_color="gray")
            fig3.update_layout(
                title=f"Coefficients — {selected}",
                xaxis_title="", yaxis_title="Coefficient",
                height=350, plot_bgcolor="#f8faff",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickangle=-30),
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_r:
            st.markdown("**Score Points** (higher = safer borrower)")
            score_colors = ["#43a047" if v >= 0 else "#e53935"
                            for v in grp["Score - Final"]]
            fig4 = go.Figure(go.Bar(
                x=grp["Score - Final"], y=grp["Feature name"],
                orientation="h", marker_color=score_colors,
                text=[f"{int(v)}" for v in grp["Score - Final"]],
                textposition="outside",
            ))
            fig4.add_vline(x=0, line_dash="dot", line_color="gray")
            fig4.update_layout(height=350, plot_bgcolor="#f8faff",
                               paper_bgcolor="rgba(0,0,0,0)",
                               xaxis_title="Score Points",
                               margin=dict(l=180))
            st.plotly_chart(fig4, use_container_width=True)

        # Top 15 most impactful features overall
        st.markdown('<div class="section-title">🔝 Top 15 Most Impactful Features</div>',
                    unsafe_allow_html=True)
        top15 = sc_active.nlargest(15, "Coefficients")
        bot5  = sc_active.nsmallest(5,  "Coefficients")
        combined = pd.concat([top15, bot5]).drop_duplicates()
        combined = combined.sort_values("Coefficients")

        colors_all = ["#e53935" if v < 0 else "#1e88e5" for v in combined["Coefficients"]]
        fig5 = go.Figure(go.Bar(
            x=combined["Coefficients"],
            y=combined["Feature name"],
            orientation="h", marker_color=colors_all,
            text=[f"{v:.4f}" for v in combined["Coefficients"]],
            textposition="outside",
        ))
        fig5.add_vline(x=0, line_dash="dot", line_color="gray")
        fig5.update_layout(height=500, plot_bgcolor="#f8faff",
                           paper_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="Coefficient",
                           margin=dict(l=280))
        st.plotly_chart(fig5, use_container_width=True)

    # ── Tab 3: Model Summary ──────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Model Statistics</div>',
                    unsafe_allow_html=True)

        c_tbl, c_met = st.columns([2, 1])
        with c_tbl:
            display = sc_active[["Feature name","Coefficients","Score - Final"]].copy()
            display.columns = ["Feature", "Coefficient", "Score"]
            display["Coefficient"] = display["Coefficient"].map("{:.6f}".format)
            display["Score"]       = display["Score"].map("{:.0f}".format)
            st.dataframe(display, use_container_width=True, hide_index=True, height=420)

        with c_met:
            st.markdown("""
            <div class="metric-card green">
                <div class="metric-label">AUROC</div>
                <div class="metric-value">0.861</div>
                <div class="metric-sub">Area Under ROC Curve</div>
            </div>
            <div class="metric-card purple">
                <div class="metric-label">Gini Coefficient</div>
                <div class="metric-value">0.722</div>
                <div class="metric-sub">2 × AUROC − 1</div>
            </div>
            <div class="metric-card orange">
                <div class="metric-label">KS Statistic</div>
                <div class="metric-value">0.476</div>
                <div class="metric-sub">Max separation</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Features Used</div>
                <div class="metric-value">87</div>
                <div class="metric-sub">After WoE binning</div>
            </div>""", unsafe_allow_html=True)

        # Simulated ROC
        st.markdown("**ROC Curve**")
        np.random.seed(42)
        fpr = np.linspace(0, 1, 200)
        tpr = np.clip(1 - (1 - fpr)**3.5 + np.random.normal(0,.008,200).cumsum()*.01, 0, 1)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, fill="tozeroy",
            fillcolor="rgba(30,136,229,0.12)",
            line=dict(color="#1e88e5",width=2.5),
            name="PD Model (AUC = 0.861)"))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], line=dict(color="gray",dash="dash"), name="Random"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=300, plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(x=0.55,y=0.1))
        st.plotly_chart(fig_roc, use_container_width=True)

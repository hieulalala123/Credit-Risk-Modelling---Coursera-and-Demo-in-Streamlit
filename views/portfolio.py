import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import generate_portfolio, risk_color


@st.cache_data
def get_portfolio(n=500):
    return generate_portfolio(n)


def render():
    st.markdown("""
    <div class="main-header">
        <h1>📁 Portfolio Analysis</h1>
        <p>Aggregate credit risk metrics across simulated loan portfolio</p>
    </div>""", unsafe_allow_html=True)

    df = get_portfolio(500)

    # ── Top KPIs ───────────────────────────────────────────────────────────────
    total_funded = df["Loan Amount"].sum()
    total_el     = df["EL"].sum()
    avg_pd       = df["PD"].mean()
    avg_lgd      = df["LGD"].mean()
    avg_ead      = df["EAD"].mean()
    el_ratio     = total_el / total_funded

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("Portfolio Size",  f"${total_funded/1e6:.2f}M", "Total funded amount", ""),
        ("Avg PD",          f"{avg_pd:.2%}",              "Mean probability of default", "red"),
        ("Avg LGD",         f"{avg_lgd:.2%}",             "Mean loss given default", "orange"),
        ("Total EL",        f"${total_el:,.0f}",          f"{el_ratio:.2%} of portfolio", "red"),
        ("Loans",           f"{len(df):,}",               "Total loan count", "green"),
    ]
    for col, (label, val, sub, color) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            col.markdown(f"""<div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts Row 1 ──────────────────────────────────────────────────────────
    c_l, c_r = st.columns(2)

    with c_l:
        st.markdown('<div class="section-title">PD Distribution</div>', unsafe_allow_html=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(x=df["PD"]*100, nbinsx=40,
                                     marker_color="#1e88e5", opacity=0.8))
        fig1.add_vline(x=avg_pd*100, line_dash="dash", line_color="#e53935",
                        annotation_text=f"Mean: {avg_pd:.2%}")
        fig1.update_layout(xaxis_title="PD (%)", yaxis_title="Count",
                            height=300, plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)

    with c_r:
        st.markdown('<div class="section-title">Risk Tier Breakdown</div>', unsafe_allow_html=True)
        risk_counts = df["Risk"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker_colors=["#43a047","#fb8c00","#e53935"],
            hole=0.42,
            textinfo="label+percent",
        ))
        fig2.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts Row 2 ──────────────────────────────────────────────────────────
    c2_l, c2_r = st.columns(2)

    with c2_l:
        st.markdown('<div class="section-title">EL by Grade</div>', unsafe_allow_html=True)
        grade_grp = df.groupby("Grade")[["EL","Loan Amount"]].sum().reset_index()
        grade_grp["EL_pct"] = grade_grp["EL"] / grade_grp["Loan Amount"] * 100
        grade_order = ["A","B","C","D","E","F","G"]
        grade_grp["Grade"] = pd.Categorical(grade_grp["Grade"], categories=grade_order, ordered=True)
        grade_grp = grade_grp.sort_values("Grade")
        colors = ["#43a047","#66bb6a","#ffee58","#ffa726","#ef5350","#c62828","#b71c1c"]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=grade_grp["Grade"], y=grade_grp["EL"],
                               marker_color=colors, name="Total EL ($)",
                               text=[f"${v:,.0f}" for v in grade_grp["EL"]],
                               textposition="outside"))
        fig3.update_layout(xaxis_title="Grade", yaxis_title="Total EL ($)",
                            height=300, plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

    with c2_r:
        st.markdown('<div class="section-title">PD vs LGD Scatter</div>', unsafe_allow_html=True)
        fig4 = px.scatter(df, x="PD", y="LGD", color="Grade",
                           size="Loan Amount", size_max=12,
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           opacity=0.6, height=300)
        fig4.update_layout(plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    # ── EL Concentration ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Expected Loss Concentration by Grade & Term</div>',
                unsafe_allow_html=True)
    pivot = df.pivot_table(values="EL", index="Grade", columns="Term", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([g for g in ["A","B","C","D","E","F","G"] if g in pivot.index])
    fig5 = go.Figure(go.Heatmap(
        x=pivot.columns, y=pivot.index,
        z=pivot.values,
        colorscale="RdYlGn_r", colorbar=dict(title="Total EL ($)"),
        text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
    ))
    fig5.update_layout(xaxis_title="Term", yaxis_title="Grade",
                        height=350, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)

    # ── Cumulative EL ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Lorenz Curve — EL Concentration</div>',
                unsafe_allow_html=True)
    sorted_el = np.sort(df["EL"].values)
    cumulative_loans = np.linspace(0, 1, len(sorted_el))
    cumulative_el    = np.cumsum(sorted_el) / sorted_el.sum()
    gini = 1 - 2 * np.trapz(cumulative_el, cumulative_loans)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=cumulative_loans, y=cumulative_el,
                               fill="tozeroy", fillcolor="rgba(30,136,229,0.15)",
                               line=dict(color="#1e88e5", width=2), name=f"Portfolio (Gini={gini:.3f})"))
    fig6.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(color="gray", dash="dash"), name="Perfect Equality"))
    fig6.update_layout(xaxis_title="Cumulative % Loans", yaxis_title="Cumulative % EL",
                        height=330, plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(x=0.02, y=0.95))
    st.plotly_chart(fig6, use_container_width=True)

    # ── Data Table ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Portfolio Sample (top 50 by EL)</div>',
                unsafe_allow_html=True)
    top50 = df.nlargest(50, "EL")[["Grade","Term","Loan Amount","Int Rate (%)","Annual Inc",
                                    "DTI","PD","LGD","EAD","EL","Risk"]]

    def color_risk(val):
        return "color: #b71c1c; font-weight:600" if val == "High" else \
               "color: #e65100" if val == "Medium" else "color: #2e7d32"

    styled = top50.style\
        .format({"PD":"{:.2%}","LGD":"{:.2%}","EAD":"${:,.0f}","EL":"${:,.2f}",
                 "Loan Amount":"${:,.0f}","Annual Inc":"${:,.0f}"})\
        .applymap(color_risk, subset=["Risk"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Full Portfolio CSV", csv,
                        file_name="credit_risk_portfolio.csv", mime="text/csv")

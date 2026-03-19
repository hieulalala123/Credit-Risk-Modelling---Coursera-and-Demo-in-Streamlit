import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model import compute_expected_loss, risk_label, risk_color


def render():
    st.markdown("""
    <div class="main-header">
        <h1>💰 Expected Loss Calculator</h1>
        <p>EL = PD × LGD × EAD — integrated credit risk metric</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 🎛️ Loan Parameters")
        grade    = st.select_slider("Grade", ["G","F","E","D","C","B","A"], value="B", key="el_gr")
        term     = st.radio("Term", [36, 60], horizontal=True, key="el_term")
        funded   = st.number_input("Loan Amount ($)", 2000, 40000, 20000, 500, key="el_fa")
        int_rate = st.slider("Interest Rate (%)", 5.0, 28.0, 11.5, 0.1, key="el_ir")
        annual_inc = st.number_input("Annual Income ($)", 10000, 300000, 80000, 5000, key="el_ai")
        dti      = st.slider("Debt-to-Income", 0.0, 45.0, 14.0, 0.5, key="el_dti")
        inq      = st.slider("Inquiries (6m)", 0, 10, 0, key="el_inq")
        open_acc = st.slider("Open Accounts", 1, 30, 10, key="el_oa")
        total_acc = st.slider("Total Accounts", 5, 60, 28, key="el_ta")
        total_rev = st.number_input("Total Rev. Hi Limit ($)", 1000, 200000, 25000, 1000, key="el_trhl")
        emp_len  = st.slider("Employment (yrs)", 0, 10, 7, key="el_el")
        months_issue = st.slider("Months Since Issue", 1, 120, 24, key="el_mi")
        months_cr    = st.slider("Months Since Earliest CR", 100, 400, 240, key="el_mc")
        hw  = st.selectbox("Home Ownership", ["MORTGAGE","RENT","OWN"], key="el_hw")
        vs  = st.selectbox("Verification", ["Source Verified","Not Verified","Verified"], key="el_vs")
        purpose = st.selectbox("Purpose", ["debt_consolidation","credit_card","home_improvement","other"], key="el_pu")

    installment = funded * (int_rate/1200) / (1 - (1 + int_rate/1200)**(-term))
    inputs = dict(
        grade=grade, term=term, funded_amnt=funded, int_rate=int_rate,
        annual_inc=annual_inc, dti=dti, inq_last_6mths=inq, open_acc=open_acc,
        total_acc=total_acc, total_rev_hi_lim=total_rev, emp_length=emp_len,
        months_since_issue_d=months_issue, months_since_earliest_cr_line=months_cr,
        home_ownership=hw, verification_status=vs, purpose=purpose,
        installment=installment,
    )

    res = compute_expected_loss(inputs)
    pd_v, lgd_v, ead_v, ccf_v, el_v = res["PD"], res["LGD"], res["EAD"], res["CCF"], res["EL"]
    rl = risk_label(pd_v)
    rc = risk_color(pd_v)
    badge_cls = {"Low":"risk-low","Medium":"risk-medium","High":"risk-high"}[rl]

    # ── Top Metrics ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Risk Components</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        ("PD",  f"{pd_v:.2%}",  f"<span class='risk-badge {badge_cls}'>{rl}</span>", rc),
        ("LGD", f"{lgd_v:.2%}", "Loss rate at default", "orange"),
        ("CCF", f"{ccf_v:.4f}", "Credit conversion factor", ""),
        ("EAD", f"${ead_v:,.0f}", f"CCF × ${funded:,.0f}", "purple"),
        ("EL",  f"${el_v:,.2f}", f"{el_v/funded:.2%} of loan", "red"),
    ]
    for col, (label, val, sub, color) in zip([c1,c2,c3,c4,c5], metrics):
        with col:
            col.markdown(f"""<div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Sankey Diagram ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 EL Flow Decomposition</div>', unsafe_allow_html=True)

    safe_amt    = funded * (1 - pd_v)
    default_amt = funded * pd_v
    recovery    = default_amt * (1 - lgd_v)
    net_loss    = default_amt * lgd_v

    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=25,
            label=["Loan Portfolio", "Expected Survival", "Expected Default",
                   "Recovery (LGD)", "Expected Loss (EL)"],
            color=["#1e88e5","#43a047","#fb8c00","#66bb6a","#e53935"],
        ),
        link=dict(
            source=[0, 0, 2, 2],
            target=[1, 2, 3, 4],
            value=[safe_amt, default_amt, recovery, net_loss],
            color=["rgba(67,160,71,0.3)","rgba(251,140,0,0.3)",
                   "rgba(102,187,106,0.3)","rgba(229,57,53,0.4)"],
            label=[f"${safe_amt:,.0f}", f"${default_amt:,.0f}",
                   f"${recovery:,.0f}", f"${net_loss:,.0f}"],
        ),
    ))
    fig_sankey.update_layout(title="Expected Loss Flow",
                              height=380, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sankey, use_container_width=True)

    # ── Sensitivity Analysis ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔬 Sensitivity Analysis</div>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**EL vs Interest Rate**")
        ir_vals = np.linspace(5, 28, 40)
        el_vals = []
        for ir in ir_vals:
            inp2 = {**inputs, "int_rate": ir}
            r2 = compute_expected_loss(inp2)
            el_vals.append(r2["EL"])
        fig_ir = go.Figure()
        fig_ir.add_trace(go.Scatter(x=ir_vals, y=el_vals, mode="lines",
                                     fill="tozeroy", fillcolor="rgba(229,57,53,0.15)",
                                     line=dict(color="#e53935", width=2)))
        fig_ir.add_vline(x=int_rate, line_dash="dash", line_color="#1e88e5",
                          annotation_text=f"Current ({int_rate}%)")
        fig_ir.update_layout(xaxis_title="Interest Rate (%)", yaxis_title="Expected Loss ($)",
                              height=280, plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ir, use_container_width=True)

    with col_s2:
        st.markdown("**EL by Grade**")
        grades = ["A","B","C","D","E","F","G"]
        el_by_grade = []
        for g in grades:
            inp3 = {**inputs, "grade": g}
            r3 = compute_expected_loss(inp3)
            el_by_grade.append(r3["EL"])
        colors = ["#43a047","#66bb6a","#ffee58","#ffa726","#ef5350","#c62828","#b71c1c"]
        fig_gr = go.Figure(go.Bar(
            x=grades, y=el_by_grade, marker_color=colors,
            text=[f"${v:,.0f}" for v in el_by_grade], textposition="outside",
        ))
        fig_gr.add_hline(y=el_v, line_dash="dash", line_color="#1e88e5",
                          annotation_text=f"Current ({grade})")
        fig_gr.update_layout(xaxis_title="Grade", yaxis_title="Expected Loss ($)",
                              height=280, plot_bgcolor="#f8faff", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gr, use_container_width=True)

    # ── EL vs DTI heatmap ─────────────────────────────────────────────────────
    st.markdown("**EL Heatmap: Interest Rate vs DTI**")
    ir_ax  = np.linspace(6, 24, 12)
    dti_ax = np.linspace(5, 40, 10)
    z = []
    for d in dti_ax:
        row = []
        for ir in ir_ax:
            inp4 = {**inputs, "int_rate": ir, "dti": d}
            row.append(compute_expected_loss(inp4)["EL"])
        z.append(row)

    fig_heat = go.Figure(go.Heatmap(
        x=[f"{v:.0f}%" for v in ir_ax],
        y=[f"{v:.0f}" for v in dti_ax],
        z=z, colorscale="RdYlGn_r",
        colorbar=dict(title="EL ($)"),
        text=[[f"${v:.0f}" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig_heat.update_layout(
        xaxis_title="Interest Rate", yaxis_title="DTI",
        height=380, paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Summary Box ────────────────────────────────────────────────────────────
    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class="info-box">
            <b>📋 Summary Report</b><br><br>
            Loan Amount: <b>${funded:,.0f}</b><br>
            Grade: <b>{grade}</b> | Term: <b>{term} months</b> | Rate: <b>{int_rate:.1f}%</b><br><br>
            PD = <b>{pd_v:.2%}</b> &nbsp;|&nbsp; LGD = <b>{lgd_v:.2%}</b> &nbsp;|&nbsp;
            EAD = <b>${ead_v:,.0f}</b><br>
            <b style="font-size:1.1rem">Expected Loss = ${el_v:,.2f} ({el_v/funded:.2%} of loan)</b><br><br>
            <span class="risk-badge {badge_cls}">{rl} Risk</span>
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="formula-box">
  EL   = PD  × LGD × EAD<br>
       = {pd:.4f} × {lgd:.4f} × ${ead:,.2f}<br>
       = ${el:,.2f}<br><br>
  CCF  = {ccf:.4f}<br>
  EAD  = CCF × ${fa:,} = ${ead:,.2f}
        </div>""".format(pd=pd_v, lgd=lgd_v, ead=ead_v, el=el_v, ccf=ccf_v, fa=funded),
        unsafe_allow_html=True)

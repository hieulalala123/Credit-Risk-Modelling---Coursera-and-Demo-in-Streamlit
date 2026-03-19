import streamlit as st


def render():

    # ── Hero Section ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0d1b2a 0%, #1e3a5f 50%, #1565c0 100%);
        padding: 3.5rem 3rem 3rem; border-radius: 20px; margin-bottom: 2rem;
        box-shadow: 0 12px 48px rgba(13,27,42,0.4); text-align: center;
    ">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">🏦</div>
        <h1 style="color: white; font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px;">
            Credit Risk Modeling
        </h1>
        <p style="color: rgba(255,255,255,0.7); font-size: 1.1rem; margin: 0.8rem 0 0; font-weight: 300;">
            Basel II / IFRS 9 compliant pipeline &nbsp;·&nbsp; LendingClub 2007–2014
        </p>
        <div style="margin-top: 1.8rem; display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
            <span style="background:rgba(255,255,255,0.12); color:white; padding:.4rem 1.1rem;
                         border-radius:50px; font-size:.85rem; font-weight:500;">📊 466K+ Loans</span>
            <span style="background:rgba(255,255,255,0.12); color:white; padding:.4rem 1.1rem;
                         border-radius:50px; font-size:.85rem; font-weight:500;">🧠 3 ML Models</span>
            <span style="background:rgba(255,255,255,0.12); color:white; padding:.4rem 1.1rem;
                         border-radius:50px; font-size:.85rem; font-weight:500;">📈 AUROC 0.861</span>
            <span style="background:rgba(255,255,255,0.12); color:white; padding:.4rem 1.1rem;
                         border-radius:50px; font-size:.85rem; font-weight:500;">⚡ Real-time EL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── About box ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#f0f7ff; border-radius:14px; padding:1.6rem 2rem;
                border-left: 5px solid #1e88e5; margin-bottom:2rem;">
        <h3 style="color:#1e3a5f; margin:0 0 .6rem; font-size:1.05rem;">🎯 Về Demo này</h3>
        <p style="color:#444; margin:0; line-height:1.8; font-size:.92rem;">
            Demo này mô phỏng một hệ thống đánh giá rủi ro tín dụng hoàn chỉnh theo chuẩn <strong>Basel II</strong>.
            Từ dữ liệu vay tiêu dùng của LendingClub, chúng tôi xây dựng 3 model độc lập —
            <strong>PD, LGD, EAD</strong> — rồi kết hợp thành chỉ số <strong>Expected Loss (EL)</strong>
            dùng để ra quyết định phê duyệt và định giá khoản vay.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔄 Modeling Pipeline</div>', unsafe_allow_html=True)

    steps = [
        ("01", "Data Preprocessing", "#3b82f6",
         "Feature engineering · WoE encoding · Binning · Missing value imputation",
         "75 raw features → clean training matrix"),
        ("02", "PD Model", "#8b5cf6",
         "Logistic Regression với p-values · WoE-encoded features · Grade / DTI / Income",
         "Output: P(default) ∈ [0,1] · AUROC = 0.861"),
        ("03", "LGD Model", "#f59e0b",
         "Two-stage: Logistic (P(recovery>0)) × Linear Regression (recovery rate)",
         "Output: Loss Given Default ∈ [0,1]"),
        ("04", "EAD Model", "#10b981",
         "Credit Conversion Factor (CCF) · Linear Regression trên defaulted loans",
         "Output: EAD = CCF × Funded Amount"),
        ("05", "Expected Loss", "#ef4444",
         "Tổng hợp 3 model · Phân tích danh mục · Stress testing · Capital allocation",
         "Output: EL = PD × LGD × EAD"),
    ]
    for num, title, color, desc, output in steps:
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:1.2rem;
                    background:white; border-radius:12px; padding:1rem 1.4rem;
                    margin-bottom:.6rem; box-shadow:0 2px 10px rgba(0,0,0,0.05);
                    border-left: 4px solid {color};">
            <div style="background:{color}; color:white; border-radius:7px;
                        padding:.25rem .6rem; font-weight:700; font-size:.72rem;
                        letter-spacing:1px; white-space:nowrap; margin-top:3px; flex-shrink:0;">
                STEP {num}
            </div>
            <div>
                <div style="font-weight:700; color:#1e3a5f; font-size:.95rem;">{title}</div>
                <div style="font-size:.80rem; color:#666; margin:.2rem 0;">{desc}</div>
                <div style="font-size:.77rem; color:{color}; font-weight:600;">→ {output}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Formula + Performance ─────────────────────────────────────────────────
    col_f, col_m = st.columns([1, 1])

    with col_f:
        st.markdown('<div class="section-title">📐 Core Formula</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#0f172a; border-radius:14px; padding:1.8rem 2rem; font-family:monospace;">
            <div style="color:#94a3b8; font-size:.72rem; margin-bottom:.8rem; letter-spacing:1.5px;">
                EXPECTED LOSS
            </div>
            <div style="color:#f1f5f9; font-size:1.6rem; font-weight:700; margin-bottom:1.2rem;">
                EL = PD × LGD × EAD
            </div>
            <div style="border-top:1px solid #1e293b; padding-top:1rem;">
                <div style="color:#60a5fa; font-size:.83rem; line-height:2.1;">
                    PD &nbsp;= P(default | borrower info)<br>
                    LGD = 1 − E[recovery rate | default]<br>
                    EAD = CCF × Funded Amount
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m:
        st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
        metrics = [
            ("PD Model",  "AUROC",    "0.861", "#3b82f6", 86.1),
            ("PD Model",  "Gini",     "0.722", "#8b5cf6", 72.2),
            ("LGD S1",    "Accuracy", "82.1%", "#f59e0b", 82.1),
            ("EAD Model", "R²",       "0.513", "#10b981", 51.3),
        ]
        for model, metric, val, color, pct in metrics:
            st.markdown(f"""
            <div style="margin-bottom:.9rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:.3rem;">
                    <span style="font-size:.82rem; color:#555;">
                        <strong style="color:#1e3a5f;">{model}</strong> · {metric}
                    </span>
                    <span style="font-weight:700; color:{color}; font-size:.88rem;">{val}</span>
                </div>
                <div style="background:#e2e8f0; border-radius:50px; height:7px;">
                    <div style="background:{color}; width:{pct}%; height:7px;
                                border-radius:50px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Dataset + Tech ────────────────────────────────────────────────────────
    col_d, col_t = st.columns(2)

    with col_d:
        st.markdown('<div class="section-title">📦 Dataset</div>', unsafe_allow_html=True)
        rows = [
            ("🗂️", "Nguồn",         "LendingClub 2007–2014"),
            ("📋", "Số records",     "466,285 khoản vay"),
            ("📌", "Features gốc",   "75 features"),
            ("🎯", "Target",         "good_bad  (0=default / 1=good)"),
            ("✂️", "Train/Test",     "80% / 20%  (random_state=12)"),
            ("❌", "Default rate",   "~20% bad loans"),
        ]
        for icon, label, val in rows:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:.5rem .4rem; border-bottom:1px solid #f1f5f9;">
                <span style="font-size:.83rem; color:#666;">{icon} {label}</span>
                <span style="font-size:.83rem; font-weight:600; color:#1e3a5f;">{val}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_t:
        st.markdown('<div class="section-title">🛠️ Tech Stack</div>', unsafe_allow_html=True)
        tech = [
            ("Python 3.10+",   "Core language",            "#3b82f6"),
            ("Scikit-learn",   "PD Logistic Regression",   "#f59e0b"),
            ("Statsmodels",    "LGD / EAD OLS + p-values", "#8b5cf6"),
            ("Pandas / NumPy", "Data wrangling",           "#10b981"),
            ("Streamlit",      "Interactive web app",      "#ef4444"),
            ("Plotly",         "Interactive charts",       "#06b6d4"),
        ]
        for name, role, color in tech:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:.9rem;
                        padding:.5rem .4rem; border-bottom:1px solid #f1f5f9;">
                <div style="width:9px; height:9px; border-radius:50%;
                            background:{color}; flex-shrink:0;"></div>
                <span style="font-size:.83rem;">
                    <strong style="color:#1e3a5f;">{name}</strong>
                    <span style="color:#888;"> · {role}</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Navigation Cards ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🚀 Khám phá Demo</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    cards = [
        ("📊", "PD Model", "#3b82f6",
         "Tính xác suất vỡ nợ theo thời gian thực. Phân tích WoE từng feature và ROC curve của model."),
        ("💰", "Expected Loss", "#ef4444",
         "Nhập thông số khoản vay → nhận ngay PD · LGD · EAD · EL. Sensitivity analysis theo lãi suất & grade."),
        ("📁", "Portfolio", "#10b981",
         "Phân tích tổng danh mục 500 khoản vay mô phỏng: Lorenz curve, heatmap rủi ro, phân bố EL."),
    ]
    for col, (icon, title, color, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div style="background:white; border-radius:14px; padding:1.5rem 1.3rem;
                        box-shadow:0 4px 18px rgba(0,0,0,0.07);
                        border-top: 4px solid {color}; text-align:center; height:100%;">
                <div style="font-size:2rem; margin-bottom:.5rem;">{icon}</div>
                <div style="font-weight:700; color:#1e3a5f; font-size:.98rem;
                            margin-bottom:.6rem;">{title}</div>
                <div style="font-size:.81rem; color:#666; line-height:1.65;">{desc}</div>
                <div style="margin-top:1.1rem;">
                    <span style="background:{color}18; color:{color}; font-size:.77rem;
                                 font-weight:600; padding:.3rem .9rem; border-radius:50px;">
                        👈 Chọn từ sidebar
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

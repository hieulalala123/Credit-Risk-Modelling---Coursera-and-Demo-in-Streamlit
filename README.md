# 🏦 Credit Risk Modeling System

> **Basel II / IFRS 9 compliant credit risk pipeline** built on LendingClub 2007–2014 data.  
> Interactive Streamlit demo covering PD, LGD, EAD, and Expected Loss modeling.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

This project implements a full credit risk modeling pipeline:

| Step | Model | Method | Output |
|------|-------|--------|--------|
| 1 | **Data Preprocessing** | WoE encoding, binning, feature engineering | Clean training data |
| 2 | **PD Model** | Logistic Regression with p-values | Probability of Default |
| 3 | **LGD Model** | Two-stage: Logistic + Linear Regression | Loss Given Default |
| 4 | **EAD Model** | Linear Regression on CCF | Exposure at Default |
| 5 | **Expected Loss** | EL = PD × LGD × EAD | Regulatory capital metric |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-modeling.git
cd credit-risk-modeling
pip install -r requirements.txt
```

### 2. Run the App

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
credit-risk-modeling/
│
├── app.py                          # Main Streamlit entry point
├── requirements.txt
├── README.md
│
├── utils/
│   ├── __init__.py
│   └── model.py                    # PD / LGD / EAD / EL computation
│
├── pages/
│   ├── __init__.py
│   ├── home.py                     # Overview & methodology
│   ├── pd_model.py                 # PD calculator + WoE analysis
│   ├── lgd_ead.py                  # LGD & EAD calculators
│   ├── el_calc.py                  # Expected Loss + sensitivity
│   └── portfolio.py                # Portfolio-level aggregation
│
└── notebooks/                      # Original Jupyter notebooks
    ├── 1_Data_Preprocessing.ipynb
    ├── 2_PD.ipynb
    ├── 3_LGD_and_EAD.ipynb
    └── 4_EL.ipynb
```

---

## 📐 Key Formulas

```
EL = PD × LGD × EAD

PD  = σ(β₀ + Σ βᵢ · WoEᵢ)           # Logistic Regression
LGD = 1 − P(rec>0) × E[rec | rec>0]  # Two-stage model
EAD = CCF × Funded Amount             # CCF via Linear Regression
```

### Weight of Evidence (WoE)
```
WoEᵢ = ln(P(Good)ᵢ / P(Bad)ᵢ)

IV = Σ (P(Good)ᵢ − P(Bad)ᵢ) × WoEᵢ
```

---

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| PD    | AUROC  | 0.861 |
| PD    | Gini   | 0.722 |
| PD    | KS Stat| 0.476 |
| LGD Stage 1 | Accuracy | ~82% |
| EAD   | R²     | ~0.51 |

---

## 📦 Dataset

**LendingClub Loan Data 2007–2014**
- ~466,000 loan records
- 75 original features
- Target: `good_bad` binary (0 = default, 1 = non-default)
- Source: [Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)

---

## 🔧 Tech Stack

- **Python** 3.10+
- **Streamlit** — interactive web app
- **Scikit-learn** — logistic / linear regression
- **Statsmodels** — OLS with p-values
- **Plotly** — interactive charts
- **Pandas / NumPy** — data wrangling

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

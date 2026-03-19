"""
Credit Risk Model Engine
Uses actual trained models: pd_model.sav, lgd_model_stage_1.sav,
lgd_model_stage_2.sav, ead.sav
"""
import os, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def _find_models_dir() -> str:
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"),
        os.path.join(os.getcwd(), "models"),
    ]
    here = os.path.abspath(__file__)
    for _ in range(4):
        here = os.path.dirname(here)
        candidates.append(os.path.join(here, "models"))
    for p in candidates:
        if p and os.path.isdir(p) and os.path.exists(os.path.join(p, "pd_model.sav")):
            return p
    return candidates[0]


_BASE = _find_models_dir()

_models = {}

def _get(name):
    if name not in _models:
        from utils.loader import load_model
        _models[name] = load_model(os.path.join(_BASE, name))
    return _models[name]


# ── PD feature list (87, matching training order) ─────────────────────────────
PD_FEATURES = [
    "grade:A","grade:B","grade:C","grade:D","grade:E","grade:F",
    "home_ownership:MORTGAGE","home_ownership:OWN",
    "verification_status:Not Verified","verification_status:Source Verified",
    "purpose:credit_card","purpose:debt_consolidation",
    "addr_state:CA","addr_state:NY","addr_state:TX",
    "initial_list_status:w",
    "addr_state:MO_HI_NC_LA","addr_state:NH_AK_MS_WV_WY_DC_ME_ID",
    "addr_state:CO_CT_SC_VT","addr_state:IL_KS","addr_state:OR_MT_WI",
    "addr_state:GA_MN_IN_WA","addr_state:DE_MA_UT_KY","addr_state:PA_SD_OH",
    "addr_state:NM_AZ_RI_MI_AR_TN",
    "purpose:vacation_major_purch__car__home_impr",
    "term:36",
    "emp_length:1-4","emp_length:5-6","emp_length:7-9","emp_length:10",
    "mths_since_issue_d:<38","mths_since_issue_d:38-39",
    "mths_since_issue_d:40-41","mths_since_issue_d:42-48",
    "mths_since_issue_d:49-52","mths_since_issue_d:53-64",
    "mths_since_issue_d:65-84",
    "int_rate:<9.548","int_rate:9.548-12.025",
    "int_rate:12.025-15.74","int_rate:15.74-20.281",
    "mths_since_earliest_cr_line:141-164","mths_since_earliest_cr_line:165-247",
    "mths_since_earliest_cr_line:248-270","mths_since_earliest_cr_line:271-352",
    "mths_since_earliest_cr_line:>352",
    "inq_last_6mths:0","inq_last_6mths:1-2","inq_last_6mths:3-6",
    "open_acc:1-3","open_acc:4-12","open_acc:13-17",
    "open_acc:18-22","open_acc:23-25","open_acc:26-30","open_acc:>=31",
    "acc_now_delinq:>=1",
    "annual_inc:20K-30K","annual_inc:30K-40K","annual_inc:40K-50K",
    "annual_inc:50K-60K","annual_inc:60K-70K","annual_inc:70K-80K",
    "annual_inc:80K-90K","annual_inc:90K-100K",
    "annual_inc:100K-125K","annual_inc:125K-150K","annual_inc:>150K",
    "mths_since_last_delinq:Missing","mths_since_last_delinq:4-30",
    "mths_since_last_delinq:31-56","mths_since_last_delinq:>=57",
    "dti:<=1.4","dti:1.4-3.5","dti:3.5-7.7","dti:7.7-10.5",
    "dti:10.5-16.1","dti:16.1-20.3","dti:20.3-21.7",
    "dti:21.7-22.4","dti:22.4-35",
    "mths_since_last_record:Missing","mths_since_last_record:3-20",
    "mths_since_last_record:21-31","mths_since_last_record:32-80",
    "mths_since_last_record:81-86",
]


def _bin_pd(inp: dict) -> dict:
    v = {f: 0 for f in PD_FEATURES}
    g  = inp.get("grade", "G")
    hw = inp.get("home_ownership", "RENT")
    vs = inp.get("verification_status", "Verified")
    pu = inp.get("purpose", "other")
    if g  in ("A","B","C","D","E","F"):           v[f"grade:{g}"] = 1
    if hw in ("MORTGAGE","OWN"):                   v[f"home_ownership:{hw}"] = 1
    if vs in ("Not Verified","Source Verified"):   v[f"verification_status:{vs}"] = 1
    if pu in ("credit_card","debt_consolidation"): v[f"purpose:{pu}"] = 1
    elif pu in ("vacation","major_purchase","car","home_improvement"):
        v["purpose:vacation_major_purch__car__home_impr"] = 1
    if inp.get("initial_list_status","f") == "w":  v["initial_list_status:w"] = 1

    # State → grouped bins
    state_map = {
        "CA":"addr_state:CA","NY":"addr_state:NY","TX":"addr_state:TX",
        **{s:"addr_state:MO_HI_NC_LA"            for s in ["MO","HI","NC","LA"]},
        **{s:"addr_state:NH_AK_MS_WV_WY_DC_ME_ID" for s in ["NH","AK","MS","WV","WY","DC","ME","ID"]},
        **{s:"addr_state:CO_CT_SC_VT"             for s in ["CO","CT","SC","VT"]},
        **{s:"addr_state:IL_KS"                   for s in ["IL","KS"]},
        **{s:"addr_state:OR_MT_WI"                for s in ["OR","MT","WI"]},
        **{s:"addr_state:GA_MN_IN_WA"             for s in ["GA","MN","IN","WA"]},
        **{s:"addr_state:DE_MA_UT_KY"             for s in ["DE","MA","UT","KY"]},
        **{s:"addr_state:PA_SD_OH"                for s in ["PA","SD","OH"]},
        **{s:"addr_state:NM_AZ_RI_MI_AR_TN"       for s in ["NM","AZ","RI","MI","AR","TN"]},
    }
    st = inp.get("addr_state","")
    if st in state_map and state_map[st] in v: v[state_map[st]] = 1

    if inp.get("term", 60) == 36: v["term:36"] = 1

    el = inp.get("emp_length", 0)
    if   1 <= el <= 4:  v["emp_length:1-4"] = 1
    elif 5 <= el <= 6:  v["emp_length:5-6"] = 1
    elif 7 <= el <= 9:  v["emp_length:7-9"] = 1
    elif el == 10:      v["emp_length:10"]   = 1

    mi = inp.get("months_since_issue_d", 0)
    if   mi < 38:           v["mths_since_issue_d:<38"]   = 1
    elif 38  <= mi <= 39:   v["mths_since_issue_d:38-39"] = 1
    elif 40  <= mi <= 41:   v["mths_since_issue_d:40-41"] = 1
    elif 42  <= mi <= 48:   v["mths_since_issue_d:42-48"] = 1
    elif 49  <= mi <= 52:   v["mths_since_issue_d:49-52"] = 1
    elif 53  <= mi <= 64:   v["mths_since_issue_d:53-64"] = 1
    elif 65  <= mi <= 84:   v["mths_since_issue_d:65-84"] = 1

    ir = inp.get("int_rate", 25.0)
    if   ir < 9.548:             v["int_rate:<9.548"]        = 1
    elif 9.548  <= ir < 12.025:  v["int_rate:9.548-12.025"]  = 1
    elif 12.025 <= ir < 15.74:   v["int_rate:12.025-15.74"]  = 1
    elif 15.74  <= ir <= 20.281: v["int_rate:15.74-20.281"]  = 1

    mc = inp.get("months_since_earliest_cr_line", 100)
    if   141 <= mc <= 164: v["mths_since_earliest_cr_line:141-164"] = 1
    elif 165 <= mc <= 247: v["mths_since_earliest_cr_line:165-247"] = 1
    elif 248 <= mc <= 270: v["mths_since_earliest_cr_line:248-270"] = 1
    elif 271 <= mc <= 352: v["mths_since_earliest_cr_line:271-352"] = 1
    elif mc > 352:         v["mths_since_earliest_cr_line:>352"]    = 1

    inq = inp.get("inq_last_6mths", 7)
    if   inq == 0:       v["inq_last_6mths:0"]   = 1
    elif 1 <= inq <= 2:  v["inq_last_6mths:1-2"] = 1
    elif 3 <= inq <= 6:  v["inq_last_6mths:3-6"] = 1

    oa = inp.get("open_acc", 0)
    if   1  <= oa <= 3:  v["open_acc:1-3"]   = 1
    elif 4  <= oa <= 12: v["open_acc:4-12"]  = 1
    elif 13 <= oa <= 17: v["open_acc:13-17"] = 1
    elif 18 <= oa <= 22: v["open_acc:18-22"] = 1
    elif 23 <= oa <= 25: v["open_acc:23-25"] = 1
    elif 26 <= oa <= 30: v["open_acc:26-30"] = 1
    elif oa >= 31:       v["open_acc:>=31"]  = 1

    if inp.get("acc_now_delinq", 0) >= 1: v["acc_now_delinq:>=1"] = 1

    ai = inp.get("annual_inc", 0)
    if   20000  <= ai < 30000:  v["annual_inc:20K-30K"]  = 1
    elif 30000  <= ai < 40000:  v["annual_inc:30K-40K"]  = 1
    elif 40000  <= ai < 50000:  v["annual_inc:40K-50K"]  = 1
    elif 50000  <= ai < 60000:  v["annual_inc:50K-60K"]  = 1
    elif 60000  <= ai < 70000:  v["annual_inc:60K-70K"]  = 1
    elif 70000  <= ai < 80000:  v["annual_inc:70K-80K"]  = 1
    elif 80000  <= ai < 90000:  v["annual_inc:80K-90K"]  = 1
    elif 90000  <= ai < 100000: v["annual_inc:90K-100K"] = 1
    elif 100000 <= ai < 125000: v["annual_inc:100K-125K"]= 1
    elif 125000 <= ai < 150000: v["annual_inc:125K-150K"]= 1
    elif ai >= 150000:          v["annual_inc:>150K"]    = 1

    mld = inp.get("mths_since_last_delinq", -1)
    if   mld < 0:          v["mths_since_last_delinq:Missing"] = 1
    elif 4  <= mld <= 30:  v["mths_since_last_delinq:4-30"]    = 1
    elif 31 <= mld <= 56:  v["mths_since_last_delinq:31-56"]   = 1
    elif mld >= 57:        v["mths_since_last_delinq:>=57"]     = 1

    dti = inp.get("dti", 36.0)
    if   dti <= 1.4:             v["dti:<=1.4"]    = 1
    elif 1.4  < dti <= 3.5:      v["dti:1.4-3.5"]  = 1
    elif 3.5  < dti <= 7.7:      v["dti:3.5-7.7"]  = 1
    elif 7.7  < dti <= 10.5:     v["dti:7.7-10.5"] = 1
    elif 10.5 < dti <= 16.1:     v["dti:10.5-16.1"]= 1
    elif 16.1 < dti <= 20.3:     v["dti:16.1-20.3"]= 1
    elif 20.3 < dti <= 21.7:     v["dti:20.3-21.7"]= 1
    elif 21.7 < dti <= 22.4:     v["dti:21.7-22.4"]= 1
    elif 22.4 < dti <= 35:       v["dti:22.4-35"]  = 1

    mlr = inp.get("mths_since_last_record", -1)
    if   mlr < 0:          v["mths_since_last_record:Missing"] = 1
    elif 3  <= mlr <= 20:  v["mths_since_last_record:3-20"]    = 1
    elif 21 <= mlr <= 31:  v["mths_since_last_record:21-31"]   = 1
    elif 32 <= mlr <= 80:  v["mths_since_last_record:32-80"]   = 1
    elif 81 <= mlr <= 86:  v["mths_since_last_record:81-86"]   = 1
    return v


def _build_lgd_s2(inp):
    g=inp.get("grade","G"); hw=inp.get("home_ownership","RENT")
    pu=inp.get("purpose","other"); ils=inp.get("initial_list_status","f")
    return np.array([[
        1 if g=="A" else 0, 1 if g=="B" else 0,
        1 if g=="C" else 0, 1 if g=="D" else 0, 1 if g=="E" else 0,
        1 if hw=="MORTGAGE" else 0, 1 if hw=="OWN" else 0,
        1 if pu=="educational" else 0, 1 if pu=="moving" else 0,
        1 if ils=="w" else 0,
        inp.get("months_since_issue_d",0),
        inp.get("months_since_earliest_cr_line",0),
        inp.get("int_rate",12.0), inp.get("inq_last_6mths",0),
        inp.get("open_acc",5), inp.get("total_acc",20),
        inp.get("total_rev_hi_lim",15000),
    ]])


def _build_ead(inp):
    g=inp.get("grade","G"); hw=inp.get("home_ownership","RENT")
    vs=inp.get("verification_status","Verified"); pu=inp.get("purpose","other")
    ils=inp.get("initial_list_status","f")
    return np.array([[
        1 if g=="A" else 0, 1 if g=="B" else 0, 1 if g=="C" else 0,
        1 if g=="D" else 0, 1 if g=="E" else 0, 1 if g=="F" else 0,
        1 if hw=="MORTGAGE" else 0,
        1 if vs=="Source Verified" else 0,
        1 if pu=="debt_consolidation" else 0, 1 if pu=="educational" else 0,
        1 if pu=="home_improvement" else 0,  1 if pu=="major_purchase" else 0,
        1 if pu=="medical" else 0, 1 if pu=="moving" else 0,
        1 if pu=="other" else 0, 1 if pu=="renewable_energy" else 0,
        1 if pu=="small_business" else 0, 1 if pu=="wedding" else 0,
        1 if ils=="w" else 0,
        inp.get("term",36), inp.get("emp_length",5),
        inp.get("months_since_issue_d",24),
        inp.get("months_since_earliest_cr_line",200),
        inp.get("funded_amnt",10000), inp.get("int_rate",12.0),
        inp.get("installment",300.0), inp.get("dti",15.0),
        inp.get("inq_last_6mths",0),
        inp.get("mths_since_last_delinq",0),
        inp.get("open_acc",5), inp.get("total_acc",20),
    ]])


# ── Public API ────────────────────────────────────────────────────────────────
def compute_pd(inp: dict) -> float:
    model = _get("pd_model.sav")
    X = np.array([[_bin_pd(inp)[f] for f in PD_FEATURES]])
    prob    = model.predict_proba(X)[0]
    classes = list(model.model.classes_)
    bad_idx = classes.index(0) if 0 in classes else 0
    return float(np.clip(prob[bad_idx], 0.0, 1.0))


def compute_lgd(inp: dict) -> float:
    lgd1 = _get("lgd_model_stage_1.sav")
    lgd2 = _get("lgd_model_stage_2.sav")
    X1 = np.array([[
        inp.get("term",36), inp.get("months_since_issue_d",24),
        inp.get("months_since_earliest_cr_line",200),
        inp.get("funded_amnt",10000), inp.get("installment",300),
        inp.get("dti",15), inp.get("mths_since_last_delinq",0),
        inp.get("mths_since_last_record",0),
        inp.get("total_acc",20), inp.get("total_rev_hi_lim",15000),
    ]])
    p_rec  = float(lgd1.predict_proba(X1)[0][1])
    rec    = float(np.clip(lgd2.predict(_build_lgd_s2(inp))[0], 0.0, 1.0))
    return float(np.clip(1.0 - p_rec * rec, 0.0, 1.0))


def compute_ead(inp: dict):
    ead_m = _get("ead.sav")
    ccf   = float(np.clip(ead_m.predict(_build_ead(inp))[0], 0.0, 1.0))
    return ccf * inp.get("funded_amnt", 10000), ccf


def compute_expected_loss(inp: dict) -> dict:
    pd_v       = compute_pd(inp)
    lgd_v      = compute_lgd(inp)
    ead_v, ccf = compute_ead(inp)
    return {"PD": pd_v, "LGD": lgd_v, "EAD": ead_v, "CCF": ccf,
            "EL": pd_v * lgd_v * ead_v, "funded_amnt": inp.get("funded_amnt", 10000)}


def risk_label(pd: float) -> str:
    if pd < 0.10: return "Low"
    if pd < 0.20: return "Medium"
    return "High"


def risk_color(pd: float) -> str:
    if pd < 0.10: return "green"
    if pd < 0.20: return "orange"
    return "red"


def get_scorecard() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_BASE, "df_scorecard.csv"))


def generate_portfolio(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    grades   = rng.choice(["A","B","C","D","E","F","G"], n, p=[.18,.22,.20,.17,.12,.07,.04])
    terms    = rng.choice([36,60], n, p=[.75,.25])
    amounts  = rng.integers(2000,35000,n).astype(float)
    int_rates= rng.uniform(5.5,25.0,n)
    ann_incs = rng.lognormal(10.8,0.5,n)
    dtis     = rng.uniform(1.0,38.0,n)
    inqs     = rng.integers(0,7,n)
    open_accs= rng.integers(1,30,n)
    total_acs= rng.integers(5,50,n)
    iss_mths = rng.integers(1,120,n)
    cr_mths  = rng.integers(100,400,n)
    emp_lens = rng.integers(0,11,n)
    tot_revs = rng.integers(2000,60000,n).astype(float)
    hws      = rng.choice(["MORTGAGE","OWN","RENT"],n,p=[.50,.12,.38])
    vss      = rng.choice(["Not Verified","Source Verified","Verified"],n,p=[.40,.35,.25])
    pus      = rng.choice(["credit_card","debt_consolidation","home_improvement","other"],n,
                           p=[.25,.45,.15,.15])
    ilss     = rng.choice(["f","w"],n,p=[.6,.4])
    mld      = rng.choice([-1,5,35,60],n,p=[.4,.2,.2,.2])
    mlr      = rng.choice([-1,10,25,50],n,p=[.7,.1,.1,.1])
    delinq   = rng.choice([0,1],n,p=[.92,.08])

    rows=[]
    for i in range(n):
        inst = amounts[i]*(int_rates[i]/1200)/(1-(1+int_rates[i]/1200)**(-terms[i]))
        inp = dict(
            grade=grades[i], term=int(terms[i]), funded_amnt=float(amounts[i]),
            int_rate=float(int_rates[i]), annual_inc=float(ann_incs[i]),
            dti=float(dtis[i]), inq_last_6mths=int(inqs[i]),
            open_acc=int(open_accs[i]), total_acc=int(total_acs[i]),
            months_since_issue_d=int(iss_mths[i]),
            months_since_earliest_cr_line=int(cr_mths[i]),
            installment=float(inst), total_rev_hi_lim=float(tot_revs[i]),
            emp_length=int(emp_lens[i]), home_ownership=hws[i],
            verification_status=vss[i], purpose=pus[i],
            initial_list_status=ilss[i],
            mths_since_last_delinq=int(mld[i]),
            mths_since_last_record=int(mlr[i]),
            acc_now_delinq=int(delinq[i]),
        )
        res = compute_expected_loss(inp)
        rows.append({
            "Grade": grades[i], "Term": f"{int(terms[i])} mo",
            "Loan Amount": int(amounts[i]),
            "Int Rate (%)": round(float(int_rates[i]),2),
            "Annual Inc": round(float(ann_incs[i])),
            "DTI": round(float(dtis[i]),1),
            "Home Ownership": hws[i],
            "PD":  round(res["PD"],4),
            "LGD": round(res["LGD"],4),
            "EAD": round(res["EAD"],2),
            "EL":  round(res["EL"],2),
            "Risk": risk_label(res["PD"]),
        })
    return pd.DataFrame(rows)

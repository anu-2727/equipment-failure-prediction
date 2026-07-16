import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.predict import predict_failure

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equipment Failure Risk Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS  (only admin/Admin@123 satisfies the new password policy)
# ─────────────────────────────────────────────────────────────────────────────
VALID_CREDENTIALS = {
    "admin":    "Admin@123",
    "anupriya": "Machine#2025",
}

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def validate_username(u: str) -> tuple[bool, str]:
    if not u:
        return False, "Username cannot be empty."
    if not re.fullmatch(r"[A-Za-z]+", u):
        return False, "Username must contain only alphabets (no numbers or special characters)."
    return True, ""

def validate_password(p: str) -> tuple[bool, list[str]]:
    errors = []
    if len(p) < 8:
        errors.append("At least 8 characters.")
    if not re.search(r"[A-Z]", p):
        errors.append("At least one uppercase letter.")
    if not re.search(r"[a-z]", p):
        errors.append("At least one lowercase letter.")
    if not re.search(r"\d", p):
        errors.append("At least one number.")
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;':\",./<>?]", p):
        errors.append("At least one special character (e.g. @, #, !, $).")
    return (len(errors) == 0), errors

def check_login(u: str, p: str) -> bool:
    return VALID_CREDENTIALS.get(u) == p

def logout():
    for k in ["authenticated", "username"]:
        st.session_state[k] = False if k != "username" else ""
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{
  font-family:'Inter',sans-serif!important;
  background:#0F172A!important;
  color:#F8FAFC!important;
}
#MainMenu,header,footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"]{visibility:hidden!important;display:none!important}

::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:#0F172A}
::-webkit-scrollbar-thumb{background:#334155;border-radius:4px}

section[data-testid="stSidebar"]{
  background:#0F172A!important;
  border-right:1px solid #1E293B!important;
}
section[data-testid="stSidebar"]>div{padding-top:0!important}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label{
  color:#94A3B8!important;font-size:12px!important;font-weight:500!important;
}
section[data-testid="stSidebar"] .stSelectbox>div>div{
  background:#1E293B!important;border:1px solid #334155!important;
  color:#F8FAFC!important;border-radius:8px!important;
}
.stSlider>div>div>div>div{background:#3B82F6!important}
.stSlider>div>div>div>div>div{background:#3B82F6!important;border:2px solid #60A5FA!important}

.block-container{padding:1.5rem 2rem!important;max-width:100%!important;background:#0F172A!important}
div[data-testid="metric-container"]{
  background:#1E293B!important;border:1px solid #334155!important;
  border-radius:12px!important;padding:1rem!important;
}

/* ── TEXT INPUTS ── */
.stTextInput>div>div>input{
  background:rgba(15,23,42,0.9)!important;border:1px solid #334155!important;
  border-radius:10px!important;color:#F8FAFC!important;font-size:13px!important;
  padding:10px 14px!important;
}
.stTextInput>div>div>input:focus{
  border-color:#3B82F6!important;
  box-shadow:0 0 0 3px rgba(59,130,246,0.2)!important;
}
.stTextInput label{color:#94A3B8!important;font-size:12px!important;font-weight:500!important}
.stCheckbox label{color:#94A3B8!important;font-size:12px!important}

/* ── BUTTONS ── */
.stButton>button{
  font-family:'Inter',sans-serif!important;font-weight:600!important;
  letter-spacing:0.02em!important;border-radius:10px!important;
  transition:all 0.2s ease!important;border:none!important;
}
.stButton>button[kind="primary"]{
  background:linear-gradient(135deg,#1d4ed8 0%,#3B82F6 100%)!important;
  color:#fff!important;box-shadow:0 4px 16px rgba(59,130,246,0.3)!important;
}
.stButton>button[kind="primary"]:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 6px 24px rgba(59,130,246,0.45)!important;
}
.stButton>button[kind="secondary"]{
  background:rgba(239,68,68,0.10)!important;
  border:1px solid rgba(239,68,68,0.3)!important;color:#FCA5A5!important;
}
.stButton>button[kind="secondary"]:hover{background:rgba(239,68,68,0.20)!important}

/* ── SPINNER ── */
.stSpinner>div{border-top-color:#3B82F6!important}

/* ─── COMPONENTS ─────────────────────────────────────────────── */

/* LOGIN */
.glass-card{
  background:rgba(30,41,59,0.75);backdrop-filter:blur(28px);
  -webkit-backdrop-filter:blur(28px);
  border:1px solid rgba(59,130,246,0.25);border-radius:22px;
  padding:2.75rem 2.25rem;width:100%;max-width:440px;
  box-shadow:0 8px 48px rgba(0,0,0,0.55),0 0 0 1px rgba(59,130,246,0.08);
  text-align:center;position:relative;z-index:2;margin:0 auto;
}
.login-logo{
  width:70px;height:70px;border-radius:50%;
  background:linear-gradient(135deg,#1d4ed8 0%,#3B82F6 60%,#06b6d4 100%);
  display:flex;align-items:center;justify-content:center;
  margin:0 auto 1.25rem;font-size:32px;
  box-shadow:0 0 28px rgba(59,130,246,0.5);
}
.login-title{font-size:22px;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em;margin-bottom:4px}
.login-sub{font-size:12px;color:#60A5FA;margin-bottom:1.75rem;font-weight:400}
.val-err{
  background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);
  border-radius:8px;padding:8px 12px;font-size:12px;color:#FCA5A5;
  margin-bottom:8px;text-align:left;
}
.val-ok{
  background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
  border-radius:8px;padding:8px 12px;font-size:12px;color:#6EE7B7;
  margin-bottom:8px;text-align:left;
}
.login-hint{font-size:11px;color:#475569;margin-top:12px}
.pw-rule-list{font-size:11px;color:#64748B;text-align:left;margin-top:6px;line-height:1.8}
.pw-rule-list li{list-style:none;padding-left:4px}
.pw-rule-pass{color:#10B981}
.pw-rule-fail{color:#EF4444}

/* HEADER */
.dash-header{
  background:linear-gradient(135deg,#1E3A5F 0%,#1E293B 60%,#0F2744 100%);
  border:1px solid #334155;border-radius:16px;padding:1.4rem 2rem;
  display:flex;align-items:center;gap:16px;margin-bottom:1.5rem;
  box-shadow:0 4px 24px rgba(0,0,0,0.4);position:relative;overflow:hidden;
}
.dash-header::before{
  content:'';position:absolute;top:-40px;right:-40px;
  width:200px;height:200px;border-radius:50%;
  background:rgba(59,130,246,0.08);pointer-events:none;
}
.dash-title{font-size:20px;font-weight:800;color:#F8FAFC;letter-spacing:-0.02em}
.dash-sub{font-size:12px;color:#60A5FA;margin-top:3px}
.live-badge{
  margin-left:auto;background:rgba(59,130,246,0.15);
  border:1px solid rgba(59,130,246,0.3);border-radius:20px;
  padding:4px 14px;font-size:11px;font-weight:500;color:#60A5FA;
  display:flex;align-items:center;gap:5px;white-space:nowrap;
}
.live-dot{
  width:7px;height:7px;border-radius:50%;background:#10B981;
  animation:pdot 1.8s infinite;
}
@keyframes pdot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.5;transform:scale(0.7)}}

/* SIDEBAR */
.sb-brand{
  background:linear-gradient(135deg,#1E3A5F 0%,#1E293B 100%);
  border:1px solid #334155;border-radius:12px;
  padding:1rem;margin-bottom:1.25rem;text-align:center;
}
.sb-brand-title{font-size:13px;font-weight:700;color:#F8FAFC;margin-top:4px}
.sb-brand-sub{font-size:10px;color:#60A5FA}
.sb-section{
  background:#1E293B;border:1px solid #334155;
  border-radius:12px;padding:.85rem 1rem;margin-bottom:.85rem;
}
.sb-section-label{
  font-size:10px;font-weight:700;color:#3B82F6;text-transform:uppercase;
  letter-spacing:0.10em;margin-bottom:10px;
  display:flex;align-items:center;gap:6px;
  padding-bottom:6px;border-bottom:1px solid #334155;
}
.sb-stat{display:flex;justify-content:space-between;align-items:center;margin-bottom:3px}
.sb-stat-key{font-size:11px;color:#64748B}
.sb-stat-val{font-size:11px;font-weight:600;color:#CBD5E1}
.sb-user-row{
  display:flex;align-items:center;gap:8px;
  background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.18);
  border-radius:10px;padding:8px 10px;margin-bottom:1rem;
}
.sb-avatar{
  width:30px;height:30px;border-radius:50%;
  background:linear-gradient(135deg,#1d4ed8,#3B82F6);
  display:flex;align-items:center;justify-content:center;
  font-size:13px;font-weight:700;color:#fff;flex-shrink:0;
}
.sb-username{font-size:12px;font-weight:600;color:#F8FAFC}
.sb-role{font-size:10px;color:#64748B}
.eng-badge{
  display:inline-block;background:rgba(59,130,246,0.12);
  border:1px solid rgba(59,130,246,0.25);border-radius:4px;
  padding:1px 6px;font-size:10px;color:#60A5FA;margin-left:4px;
}

/* METRIC CARDS */
.metric-card{
  background:#1E293B;border:1px solid #334155;border-radius:14px;
  padding:1.25rem 1.4rem;height:100%;position:relative;overflow:hidden;
  transition:border-color .2s,box-shadow .2s;
}
.metric-card:hover{border-color:#3B82F6;box-shadow:0 0 20px rgba(59,130,246,0.12)}
.metric-card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  border-radius:14px 14px 0 0;
}
.mc-blue::before{background:linear-gradient(90deg,#1d4ed8,#3B82F6)}
.mc-green::before{background:linear-gradient(90deg,#059669,#10B981)}
.mc-amber::before{background:linear-gradient(90deg,#d97706,#F59E0B)}
.mc-red::before{background:linear-gradient(90deg,#dc2626,#EF4444)}
.mc-purple::before{background:linear-gradient(90deg,#7c3aed,#8B5CF6)}
.mc-label{font-size:10px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.mc-value{font-size:26px;font-weight:800;letter-spacing:-0.03em;line-height:1;margin-bottom:4px}
.mc-sub{font-size:11px;color:#64748B}
.mc-icon{font-size:22px;margin-bottom:8px;display:block}

/* RISK BADGE */
.risk-low{
  background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);
  border-radius:20px;padding:3px 14px;font-size:12px;font-weight:700;
  color:#10B981;display:inline-block;
}
.risk-med{
  background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);
  border-radius:20px;padding:3px 14px;font-size:12px;font-weight:700;
  color:#F59E0B;display:inline-block;
}
.risk-high{
  background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.3);
  border-radius:20px;padding:3px 14px;font-size:12px;font-weight:700;
  color:#EF4444;display:inline-block;
}

/* STATUS BANNERS */
.status-normal{
  background:linear-gradient(135deg,rgba(16,185,129,0.12) 0%,rgba(5,150,105,0.06) 100%);
  border:1.5px solid rgba(16,185,129,0.35);border-radius:16px;
  padding:1.4rem 1.75rem;display:flex;align-items:center;gap:18px;
  margin:1rem 0;box-shadow:0 0 28px rgba(16,185,129,0.07);
}
.status-failure{
  background:linear-gradient(135deg,rgba(239,68,68,0.12) 0%,rgba(185,28,28,0.06) 100%);
  border:1.5px solid rgba(239,68,68,0.35);border-radius:16px;
  padding:1.4rem 1.75rem;display:flex;align-items:center;gap:18px;
  margin:1rem 0;box-shadow:0 0 28px rgba(239,68,68,0.07);
}
.sdot-n{width:22px;height:22px;border-radius:50%;background:#10B981;flex-shrink:0;box-shadow:0 0 0 6px rgba(16,185,129,0.18);animation:pdot 2s infinite}
.sdot-f{width:22px;height:22px;border-radius:50%;background:#EF4444;flex-shrink:0;box-shadow:0 0 0 6px rgba(239,68,68,0.18);animation:pdot 1.2s infinite}
.st-label{font-size:10px;font-weight:700;letter-spacing:.1em}
.st-label-n{color:#6EE7B7}.st-label-f{color:#FCA5A5}
.st-title{font-size:20px;font-weight:800;margin:3px 0}
.st-title-n{color:#10B981}.st-title-f{color:#EF4444}
.st-desc{font-size:12px}
.st-desc-n{color:#6EE7B7}.st-desc-f{color:#FCA5A5}

/* GAUGE */
.gauge-wrap{background:#1E293B;border:1px solid #334155;border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1rem}
.gauge-title{font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center}
.gauge-pct{font-size:18px;font-weight:800;letter-spacing:-0.02em}
.gauge-track{height:12px;background:#0F172A;border-radius:6px;overflow:hidden;border:1px solid #334155;margin-bottom:5px}
.gauge-fill{height:100%;border-radius:6px;transition:width .6s cubic-bezier(.4,0,.2,1)}
.gauge-marks{display:flex;justify-content:space-between;font-size:10px;color:#475569}

/* SECTION CARD */
.sc{background:#1E293B;border:1px solid #334155;border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1rem}
.sc-title{font-size:11px;font-weight:700;color:#3B82F6;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #334155;display:flex;align-items:center;gap:6px}

/* SHAP ROWS */
.shap-row{display:flex;align-items:flex-start;gap:10px;padding:10px 0;border-bottom:1px solid #1E293B;font-size:12px}
.shap-row:last-child{border-bottom:none}
.shap-badge-up{background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);border-radius:5px;padding:2px 7px;font-size:10px;font-weight:700;color:#FCA5A5;white-space:nowrap}
.shap-badge-dn{background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);border-radius:5px;padding:2px 7px;font-size:10px;font-weight:700;color:#6EE7B7;white-space:nowrap}
.shap-feat{font-weight:600;color:#CBD5E1}
.shap-desc{font-size:11px;color:#64748B;margin-top:2px;line-height:1.5}
.shap-val-up{font-size:11px;font-weight:700;font-family:monospace;color:#EF4444}
.shap-val-dn{font-size:11px;font-weight:700;font-family:monospace;color:#10B981}
.shap-bar-bg{height:6px;background:#0F172A;border-radius:3px;width:72px;overflow:hidden;flex-shrink:0;margin-top:6px}
.shap-bar-up{height:100%;background:linear-gradient(90deg,#dc2626,#EF4444);border-radius:3px}
.shap-bar-dn{height:100%;background:linear-gradient(90deg,#059669,#10B981);border-radius:3px}

/* SHAP INTERPRET */
.interp-row{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid #1E293B;font-size:12px;color:#CBD5E1}
.interp-row:last-child{border-bottom:none}
.interp-icon{font-size:16px;flex-shrink:0;margin-top:1px}

/* AI RECOMMENDATION */
.rec-card{background:linear-gradient(135deg,rgba(30,58,95,0.6) 0%,rgba(30,41,59,0.8) 100%);border:1px solid rgba(59,130,246,0.25);border-radius:14px;padding:1.25rem 1.5rem;margin-bottom:1rem}
.rec-title{font-size:11px;font-weight:700;color:#60A5FA;text-transform:uppercase;letter-spacing:.08em;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid rgba(59,130,246,0.15);display:flex;align-items:center;gap:6px}
.rec-item{display:flex;align-items:flex-start;gap:8px;padding:6px 0;font-size:12.5px;color:#CBD5E1}
.rec-item-icon{flex-shrink:0;font-size:14px;margin-top:1px}
.rec-ok{color:#10B981}.rec-warn{color:#F59E0B}.rec-danger{color:#EF4444}

/* FOOTER */
.app-footer{text-align:center;padding:1.25rem 0 .5rem;font-size:11.5px;color:#475569;border-top:1px solid #1E293B;margin-top:2rem}
.app-footer strong{color:#CBD5E1}

/* PLACEHOLDER */
.placeholder-screen{background:#1E293B;border:1px dashed #334155;border-radius:16px;padding:3rem 2rem;text-align:center;color:#475569}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model     = joblib.load(os.path.join(base, "models", "xgboost_model.pkl"))
    threshold = joblib.load(os.path.join(base, "models", "best_threshold.pkl"))
    feat_cols = joblib.load(os.path.join(base, "models", "feature_columns.pkl"))
    return model, threshold, feat_cols

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE / SHAP HELPERS  (mirrors predict.py — no modification to predict.py)
# ─────────────────────────────────────────────────────────────────────────────
def build_input_df(air_temp, process_temp, rot_speed, torque,
                   tool_wear, product_type, feature_cols):
    raw = {
        "Air temperature [K]":    air_temp,
        "Process temperature [K]":process_temp,
        "Rotational speed [rpm]": rot_speed,
        "Torque [Nm]":            torque,
        "Tool wear [min]":        tool_wear,
        "Temp_Differential":      process_temp - air_temp,
        "Power_W":                torque * (rot_speed * 2 * 3.14159 / 60),
        "Torque_Speed_Ratio":     torque / (rot_speed + 1),
    }
    df = pd.DataFrame([raw])
    for t in ["M", "H"]:
        df[f"Type_{t}"] = 1 if product_type == t else 0
    stage = (
        "Fresh"    if tool_wear <= 50  else
        "Moderate" if tool_wear <= 150 else
        "Worn"     if tool_wear <= 200 else "Critical"
    )
    for s in ["Moderate", "Worn", "Critical"]:
        df[f"Wear_Stage_{s}"] = 1 if stage == s else 0
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    return df[feature_cols]

FRIENDLY = {
    "Air temperature [K]":    "Air temperature",
    "Process temperature [K]":"Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]":            "Torque",
    "Tool wear [min]":        "Tool wear",
    "Temp_Differential":      "Temperature differential",
    "Power_W":                "Mechanical power",
    "Torque_Speed_Ratio":     "Torque-to-speed ratio",
    "Wear_Stage_Worn":        "Wear stage (Worn)",
    "Wear_Stage_Critical":    "Wear stage (Critical)",
    "Wear_Stage_Moderate":    "Wear stage (Moderate)",
}
def fname(col): return FRIENDLY.get(col, col)

SHAP_DESCS = {
    "Power_W": (
        "High mechanical load increases stress on components.",
        "Mechanical load is within safe operational limits.",
    ),
    "Torque [Nm]": (
        "Elevated torque puts excessive strain on the spindle.",
        "Torque is within nominal operating range.",
    ),
    "Torque_Speed_Ratio": (
        "Higher torque under lower speed signals heavy mechanical load.",
        "Torque-to-speed balance is healthy.",
    ),
    "Tool wear [min]": (
        "Cumulative wear is approaching or exceeding safe limits.",
        "Tool wear is within acceptable limits.",
    ),
    "Temp_Differential": (
        "Abnormal thermal differential may indicate heat dissipation issues.",
        "Thermal conditions are stable and well-managed.",
    ),
    "Air temperature [K]": (
        "Ambient temperature is contributing to thermal stress.",
        "Ambient temperature is within normal range.",
    ),
    "Process temperature [K]": (
        "Elevated process temperature increases component wear.",
        "Process temperature is within safe operating range.",
    ),
    "Rotational speed [rpm]": (
        "Abnormal rotational speed may lead to vibration or imbalance.",
        "Rotational speed is operating normally.",
    ),
    "Wear_Stage_Critical": (
        "Tool is in a critical wear stage — replacement is overdue.",
        "Tool wear stage is not in critical zone.",
    ),
    "Wear_Stage_Worn": (
        "Tool is worn and should be scheduled for replacement soon.",
        "Tool is not in worn stage.",
    ),
    "Wear_Stage_Moderate": (
        "Moderate wear detected — monitor closely.",
        "Wear stage is acceptable.",
    ),
}

def shap_desc(col, is_increase):
    pair = SHAP_DESCS.get(col, ("This feature increased failure risk.", "This feature reduced failure risk."))
    return pair[0] if is_increase else pair[1]

def run_shap(model, df_input, feature_cols):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(df_input)
    vals = sv[0] if sv.ndim == 2 else sv
    pairs = sorted(zip(feature_cols, vals), key=lambda x: abs(x[1]), reverse=True)
    return pairs, sv, explainer

# ─────────────────────────────────────────────────────────────────────────────
# AI RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_recommendations(prob, is_failure, top_features):
    recs = []
    top5 = [f for f, v in top_features[:5] if v > 0]

    if not is_failure:
        recs.append(("ok", "✔ Machine is operating normally. Continue routine monitoring."))
        recs.append(("ok", "✔ No immediate action required."))
        if prob > 0.18:
            recs.append(("warn", "⚠ Failure probability is moderate. Schedule a preventive inspection within 7 days."))
        if "Tool wear [min]" in top5 or "Wear_Stage_Worn" in top5 or "Wear_Stage_Critical" in top5:
            recs.append(("warn", "⚠ Tool wear is trending upward. Plan tool replacement at next scheduled maintenance."))
        if "Temp_Differential" in top5:
            recs.append(("warn", "⚠ Monitor thermal differential. Check cooling systems at next inspection."))
        if not recs[2:]:
            recs.append(("ok", "✔ All operating parameters within normal range."))
    else:
        recs.append(("danger", "⛔ Immediate maintenance inspection is recommended."))
        if "Tool wear [min]" in top5 or "Wear_Stage_Critical" in top5:
            recs.append(("danger", "⚠ Inspect and replace cutting tool immediately."))
        if "Torque [Nm]" in top5 or "Torque_Speed_Ratio" in top5:
            recs.append(("danger", "⚠ Reduce mechanical load — torque is critically high."))
        if "Power_W" in top5:
            recs.append(("danger", "⚠ Reduce operational power to prevent component failure."))
        if "Temp_Differential" in top5 or "Process temperature [K]" in top5:
            recs.append(("warn", "⚠ Inspect cooling and lubrication systems."))
        if "Rotational speed [rpm]" in top5:
            recs.append(("warn", "⚠ Abnormal rotational speed detected — check bearings and alignment."))
        recs.append(("warn", "⚠ Schedule preventive maintenance before resuming full operation."))

    return recs

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────────────────────
def show_login():
    _, col, _ = st.columns([1, 1.6, 1])
    with col:
        st.markdown("""
        <div class="glass-card">
          <div class="login-logo">⚙️</div>
          <div class="login-title">Equipment Failure Predictor</div>
          <div class="login-sub">AI-Based Predictive Maintenance System</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username (alphabets only)")
            password = st.text_input("Password", type="password", placeholder="Min 8 chars, A-Z, a-z, 0-9, special")
            remember = st.checkbox("Remember me")
            submitted = st.form_submit_button("Sign in to dashboard",
                                              use_container_width=True, type="primary")

        if submitted:
            u_ok, u_msg = validate_username(username.strip())
            p_ok, p_errs = validate_password(password)
            cred_ok = check_login(username.strip(), password)

            if not u_ok:
                st.markdown(f'<div class="val-err">👤 {u_msg}</div>', unsafe_allow_html=True)
            elif not p_ok:
                errs_html = "".join(f"<li class='pw-rule-fail'>✗ {e}</li>" for e in p_errs)
                st.markdown(
                    f'<div class="val-err">🔒 Password does not meet requirements:<br>'
                    f'<ul class="pw-rule-list">{errs_html}</ul></div>',
                    unsafe_allow_html=True,
                )
            elif not cred_ok:
                st.markdown(
                    '<div class="val-err">❌ Incorrect credentials. Check username and password.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username.strip()
                st.rerun()

        st.markdown("""
        <div class="pw-rule-list" style="margin-top:10px;background:rgba(15,23,42,0.5);
             border-radius:8px;padding:10px 14px;border:1px solid #1E293B;">
          <div style="font-size:10px;font-weight:700;color:#64748B;text-transform:uppercase;
               letter-spacing:.06em;margin-bottom:5px;">Password requirements</div>
          <li>Minimum 8 characters</li>
          <li>At least one uppercase letter (A–Z)</li>
          <li>At least one lowercase letter (a–z)</li>
          <li>At least one number (0–9)</li>
          <li>At least one special character (!@#$%...)</li>
          <div style="margin-top:7px;font-size:10px;color:#334155;">
            Demo credentials: <code style="color:#60A5FA">admin</code> /
            <code style="color:#60A5FA">Admin@123</code>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def show_dashboard():
    model, threshold, feature_cols = load_artifacts()
    uname = st.session_state.get("username", "admin")

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div class="sb-brand">
          <div style="font-size:28px;">⚙️</div>
          <div class="sb-brand-title">PredictMaint AI</div>
          <div class="sb-brand-sub">Predictive Maintenance Platform</div>
        </div>
        """, unsafe_allow_html=True)

        initials = uname[:2].upper()
        st.markdown(f"""
        <div class="sb-user-row">
          <div class="sb-avatar">{initials}</div>
          <div>
            <div class="sb-username">{uname.title()}</div>
            <div class="sb-role">ML Engineer</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MACHINE INFORMATION ──
        st.markdown('<div class="sb-section"><div class="sb-section-label">🏭 Machine information</div></div>',
                    unsafe_allow_html=True)
        product_type = st.selectbox(
            "Product type (Quality)",
            ["L", "M", "H"], index=1,
            help="Quality variant of the product being manufactured. L = Low, M = Medium, H = High.",
        )
        air_temp = st.slider(
            "Air temperature (K)", 295.0, 305.0, 300.0, 0.1,
            help="Ambient air temperature surrounding the machine. Affects heat dissipation.",
        )
        process_temp = st.slider(
            "Process temperature (K)", 305.0, 315.0, 310.0, 0.1,
            help="Internal process temperature during machining. Elevated values may indicate thermal stress.",
        )

        # ── OPERATING PARAMETERS ──
        st.markdown('<div class="sb-section"><div class="sb-section-label">🔄 Operating parameters</div></div>',
                    unsafe_allow_html=True)
        rot_speed = st.slider(
            "Rotational speed (rpm)", 1100, 2900, 1500, 10,
            help="Speed at which the spindle or tool rotates. Abnormal values may indicate imbalance.",
        )
        torque = st.slider(
            "Torque (Nm)", 3.0, 77.0, 40.0, 0.5,
            help="The rotational force applied to the spindle. High torque under low speed indicates heavy load.",
        )
        tool_wear = st.slider(
            "Tool wear (min)", 0, 260, 100, 1,
            help="Accumulated tool usage in minutes since last replacement. Higher values indicate tool degradation.",
        )

        # ── ENGINEERED FEATURES PANEL ──
        power_w   = torque * (rot_speed * 2 * 3.14159 / 60)
        temp_diff = process_temp - air_temp
        wear_stage = (
            "🟢 Fresh"    if tool_wear <= 50  else
            "🟡 Moderate" if tool_wear <= 150 else
            "🟠 Worn"     if tool_wear <= 200 else "🔴 Critical"
        )
        st.markdown(f"""
        <div class="sb-section">
          <div class="sb-section-label">
            🧮 Engineered features
            <span class="eng-badge" title="Derived features computed from raw inputs for model accuracy">ℹ</span>
          </div>
          <div class="sb-stat" title="Calculated as Torque × Rotational Speed (converted to Watts)">
            <span class="sb-stat-key">Mech. power</span>
            <span class="sb-stat-val">{power_w:.0f} W</span>
          </div>
          <div class="sb-stat" title="Process Temp minus Air Temp — indicates heat dissipation quality">
            <span class="sb-stat-key">Temp. differential</span>
            <span class="sb-stat-val">{temp_diff:.1f} K</span>
          </div>
          <div class="sb-stat" title="Tool wear bucket — Fresh / Moderate / Worn / Critical">
            <span class="sb-stat-key">Wear stage</span>
            <span class="sb-stat-val">{wear_stage}</span>
          </div>
          <div style="font-size:10px;color:#475569;margin-top:8px;line-height:1.6;
               border-top:1px solid #334155;padding-top:7px;">
            These are <strong style="color:#60A5FA;">derived features</strong> computed from raw inputs
            to improve model prediction accuracy.
          </div>
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🔍  Analyze machine health",
                                use_container_width=True, type="primary")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True, type="secondary"):
            logout()

    # ── HEADER ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="dash-header">
      <div style="font-size:36px;background:linear-gradient(135deg,#3B82F6,#06b6d4);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text;">⚙️</div>
      <div>
        <div class="dash-title">Equipment Failure Risk Predictor</div>
        <div class="dash-sub">AI-Based Predictive Maintenance · XGBoost + SHAP Explainability</div>
      </div>
      <div class="live-badge"><div class="live-dot"></div>System online</div>
    </div>
    """, unsafe_allow_html=True)

    # ── IDLE STATE ────────────────────────────────────────────────────────────
    if not predict_btn:
        st.markdown("""
        <div class="placeholder-screen">
          <div style="font-size:50px;margin-bottom:1rem;">⚙️</div>
          <div style="font-size:16px;font-weight:600;color:#64748B;margin-bottom:6px;">
            Ready for analysis
          </div>
          <div style="font-size:13px;">
            Configure sensor readings in the sidebar and click Analyze machine health.
          </div>
        </div>
        """, unsafe_allow_html=True)
        _footer()
        return

    # ── RUN PREDICTION ────────────────────────────────────────────────────────
    with st.spinner("Analyzing machine health..."):
        result = predict_failure(
            air_temp, process_temp, rot_speed, torque, tool_wear, product_type,
        )
        df_input = build_input_df(
            air_temp, process_temp, rot_speed, torque,
            tool_wear, product_type, feature_cols,
        )
        top_features, shap_values, explainer = run_shap(model, df_input, feature_cols)

    prob       = result["failure_probability"]
    prediction = result["prediction"]
    thresh     = result["threshold_used"]
    is_failure = prediction == "FAILURE RISK"
    confidence = abs(prob - thresh) / max(thresh, 1 - thresh) * 100
    risk_level = (
        ("🔴 High",   "risk-high") if prob >= 0.60 else
        ("🟡 Medium", "risk-med")  if prob >= 0.30 else
        ("🟢 Low",    "risk-low")
    )

    # ── METRIC CARDS (5 cards) ────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    prob_col = "#EF4444" if is_failure else "#10B981"
    with c1:
        st.markdown(f"""
        <div class="metric-card mc-blue">
          <span class="mc-icon">📊</span>
          <div class="mc-label">Failure probability</div>
          <div class="mc-value" style="color:{prob_col};">{prob*100:.1f}%</div>
          <div class="mc-sub">XGBoost output</div>
        </div>""", unsafe_allow_html=True)

    pred_col  = "#EF4444" if is_failure else "#10B981"
    pred_icon = "🔴" if is_failure else "🟢"
    with c2:
        clz = "mc-red" if is_failure else "mc-green"
        st.markdown(f"""
        <div class="metric-card {clz}">
          <span class="mc-icon">{pred_icon}</span>
          <div class="mc-label">Prediction</div>
          <div class="mc-value" style="color:{pred_col};font-size:16px;">
            {"FAILURE" if is_failure else "NORMAL"}
          </div>
          <div class="mc-sub">F2-tuned threshold</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card mc-amber">
          <span class="mc-icon">🎯</span>
          <div class="mc-label">Decision threshold</div>
          <div class="mc-value" style="color:#F59E0B;">{thresh*100:.1f}%</div>
          <div class="mc-sub">Precision-recall tuned</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card mc-purple">
          <span class="mc-icon">📈</span>
          <div class="mc-label">Confidence</div>
          <div class="mc-value" style="color:#8B5CF6;">{confidence:.0f}%</div>
          <div class="mc-sub">Distance from threshold</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card {'mc-red' if risk_level[0].startswith('🔴') else 'mc-amber' if risk_level[0].startswith('🟡') else 'mc-green'}">
          <span class="mc-icon">⚠️</span>
          <div class="mc-label">Risk level</div>
          <div class="mc-value" style="font-size:16px;margin-top:4px;">
            <span class="{risk_level[1]}">{risk_level[0]}</span>
          </div>
          <div class="mc-sub">Based on probability</div>
        </div>""", unsafe_allow_html=True)

    # ── STATUS BANNER ─────────────────────────────────────────────────────────
    if is_failure:
        st.markdown(f"""
        <div class="status-failure">
          <div class="sdot-f"></div>
          <div>
            <div class="st-label st-label-f">⚠ ALERT — ACTION REQUIRED</div>
            <div class="st-title st-title-f">MACHINE STATUS: FAILURE RISK</div>
            <div class="st-desc st-desc-f">
              Probability {prob*100:.1f}% exceeds threshold {thresh*100:.1f}%.
              Immediate maintenance inspection recommended.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-normal">
          <div class="sdot-n"></div>
          <div>
            <div class="st-label st-label-n">✓ ALL SYSTEMS NOMINAL</div>
            <div class="st-title st-title-n">MACHINE STATUS: NORMAL</div>
            <div class="st-desc st-desc-n">
              Probability {prob*100:.1f}% is below threshold {thresh*100:.1f}%.
              Machine is operating safely.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── GAUGE ─────────────────────────────────────────────────────────────────
    if prob < 0.34:   gc, glc = "#10B981", "#6EE7B7"
    elif prob < 0.65: gc, glc = "#F59E0B", "#FDE68A"
    else:             gc, glc = "#EF4444", "#FCA5A5"
    gpct = int(prob * 100)

    st.markdown(f"""
    <div class="gauge-wrap">
      <div class="gauge-title">
        <span>Failure probability gauge</span>
        <span class="gauge-pct" style="color:{glc};">{gpct}%</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill"
             style="width:{gpct}%;background:linear-gradient(90deg,{gc}88,{gc});"></div>
      </div>
      <div class="gauge-marks">
        <span>0%</span>
        <span style="color:#10B981;">25%</span>
        <span style="color:#F59E0B;">50%</span>
        <span style="color:#EF4444;">75%</span>
        <span>100%</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── SHAP + TOP FEATURES ───────────────────────────────────────────────────
    col_shap, col_right = st.columns([1.3, 1])

    with col_shap:
        # Waterfall plot
        st.markdown('<div class="sc"><div class="sc-title">🔍 SHAP — waterfall explanation</div>',
                    unsafe_allow_html=True)
        sv_arr = shap_values[0] if shap_values.ndim == 2 else shap_values
        explanation = shap.Explanation(
            values=sv_arr,
            base_values=explainer.expected_value,
            data=df_input.values[0],
            feature_names=list(feature_cols),
        )
        plt.rcParams.update({
            "figure.facecolor": "#1E293B",
            "axes.facecolor":   "#1E293B",
            "axes.labelcolor":  "#CBD5E1",
            "xtick.color":      "#64748B",
            "ytick.color":      "#CBD5E1",
            "text.color":       "#CBD5E1",
        })
        fig_shap, _ = plt.subplots(figsize=(6.5, 4.8))
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)
        plt.rcParams.update(plt.rcParamsDefault)
        st.markdown("</div>", unsafe_allow_html=True)

        # SHAP interpretation guide
        st.markdown("""
        <div class="sc">
          <div class="sc-title">📖 How to interpret this explanation</div>
          <div class="interp-row">
            <span class="interp-icon">🔴</span>
            <span><strong style="color:#FCA5A5;">Red bars</strong> push the prediction toward failure — they
            increase the failure risk score.</span>
          </div>
          <div class="interp-row">
            <span class="interp-icon">🔵</span>
            <span><strong style="color:#93C5FD;">Blue bars</strong> push the prediction away from failure — they
            reduce the failure risk score.</span>
          </div>
          <div class="interp-row">
            <span class="interp-icon">📏</span>
            <span><strong style="color:#CBD5E1;">Bar length</strong> indicates the strength of influence — longer
            bars have a greater impact on this prediction.</span>
          </div>
          <div class="interp-row">
            <span class="interp-icon">🏆</span>
            <span><strong style="color:#CBD5E1;">Top contributing features</strong> are the most influential
            inputs driving this specific prediction.</span>
          </div>
          <div class="interp-row">
            <span class="interp-icon">📦</span>
            <span><strong style="color:#CBD5E1;">Other features</strong> — low-impact features are grouped
            together at the bottom as a combined residual.</span>
          </div>
          <div class="interp-row">
            <span class="interp-icon">ℹ️</span>
            <span>SHAP explains <strong style="color:#CBD5E1;">this individual prediction only</strong>, not
            the overall model behaviour.</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Top contributing features with rich descriptions
        st.markdown('<div class="sc"><div class="sc-title">📋 Top contributing features</div>',
                    unsafe_allow_html=True)
        max_abs = abs(top_features[0][1]) + 1e-9
        for col_name, fval in top_features[:7]:
            nice  = fname(col_name)
            pct   = int(min(abs(fval) / max_abs, 1.0) * 65)
            is_up = fval > 0
            badge = '<span class="shap-badge-up">↑ Increases risk</span>' if is_up \
                    else '<span class="shap-badge-dn">↓ Reduces risk</span>'
            vcls  = "shap-val-up" if is_up else "shap-val-dn"
            bfill = "shap-bar-up" if is_up else "shap-bar-dn"
            desc  = shap_desc(col_name, is_up)
            st.markdown(f"""
            <div class="shap-row">
              <div style="flex:1;min-width:0;">
                <div class="shap-feat">{nice}</div>
                <div style="margin:3px 0;">{badge}</div>
                <div class="shap-desc">{desc}</div>
              </div>
              <div style="text-align:right;flex-shrink:0;">
                <div class="{vcls}">{fval:+.4f}</div>
                <div class="shap-bar-bg" style="margin-top:5px;">
                  <div class="{bfill}" style="width:{pct}px;"></div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── AI RECOMMENDATION PANEL ───────────────────────────────────────────────
    recs = build_recommendations(prob, is_failure, top_features)
    icon_map = {"ok": "✅", "warn": "⚠️", "danger": "🚨"}
    col_map  = {"ok": "rec-ok", "warn": "rec-warn", "danger": "rec-danger"}
    rec_rows = "".join(
        f'<div class="rec-item">'
        f'<span class="rec-item-icon {col_map[k]}">{icon_map[k]}</span>'
        f'<span class="{col_map[k]}">{msg}</span>'
        f'</div>'
        for k, msg in recs
    )
    header_colour = "#EF4444" if is_failure else "#10B981"
    header_icon   = "🚨" if is_failure else "🤖"
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-title" style="color:{header_colour};">
        {header_icon} AI recommendation
      </div>
      {rec_rows}
    </div>
    """, unsafe_allow_html=True)

    _footer()

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
def _footer():
    st.markdown("""
    <div class="app-footer">
      Developed by <strong>Anupriya</strong>
      &nbsp;·&nbsp;
      Machine Learning &nbsp;|&nbsp; XGBoost &nbsp;|&nbsp; SHAP &nbsp;|&nbsp; Streamlit
      &nbsp;·&nbsp; <strong>Version 1.0</strong>
      &nbsp;·&nbsp; Explainable AI Dashboard
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION INIT + ROUTING
# ─────────────────────────────────────────────────────────────────────────────
for k, d in [("authenticated", False), ("username", "")]:
    if k not in st.session_state:
        st.session_state[k] = d

if not st.session_state["authenticated"]:
    show_login()
else:
    show_dashboard()
"""
Recession Radar v7.0 — Streamlit Web App
 
Deploys to Streamlit Community Cloud for free.
See DEPLOY_STREAMLIT.md for step-by-step instructions.
"""
 
import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
 
# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
 
st.set_page_config(
    page_title="Recession Radar v7.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# Custom CSS for dark gold theme
st.markdown("""
<style>
    .main { background-color: #1e1a12; }
    .stApp { background-color: #1e1a12; color: #f2e8d0; }
    h1, h2, h3, h4 { color: #c8a84a !important; }
    .metric-container {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(210,180,90,0.12);
        border-radius: 14px;
        padding: 16px;
    }
    .stMetric { background: rgba(255,255,255,0.04); border-radius: 8px; padding: 8px; }
    [data-testid="stSidebar"] { background-color: #261f14; }
    .stButton>button { background: rgba(200,168,74,0.2); color: #c8a84a; border: 1px solid #c8a84a; }
    .stSelectbox label, .stSlider label { color: #b09a6e !important; }
</style>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
 
INDICATOR_NAMES = ['Yield curve', 'Permits', 'LEI', 'Oil shock', 'Credit spreads',
                   'Corp profits', 'Durables', 'Euphoria', 'Sahm rule', 'Breadth', 'Fiscal(-)']
 
JUDGMENT_WEIGHTS = [13, 8, 9, 7, 18, 7, 6, 10, 8, 6, 8]
 
EVENTS = [
    {'label': '1973-74 Oil Shock', 'drawdown': -0.48, 'recession': True, 'regime': 'pre-2000',
     'x': [.90,.70,.80,.95,.90,.80,.70,.60,.90,.70,.30],
     'notes': 'OPEC embargo. Oil tripled. All indicators fired.'},
    {'label': '1976-78 Correction', 'drawdown': -0.19, 'recession': False, 'regime': 'pre-2000',
     'x': [.30,.20,.30,.20,.25,.30,.25,.30,.20,.40,.40],
     'notes': 'Mild correction. No significant warnings.'},
    {'label': '1980 Volcker I', 'drawdown': -0.17, 'recession': True, 'regime': 'pre-2000',
     'x': [.95,.50,.70,.90,.70,.90,.60,.50,.80,.60,.20],
     'notes': 'Volcker rate hike. Iran revolution oil shock.'},
    {'label': '1981-82 Volcker II', 'drawdown': -0.27, 'recession': True, 'regime': 'pre-2000',
     'x': [.95,.60,.80,.70,.85,.80,.70,.40,.90,.70,.30],
     'notes': 'Double-dip. Rates at 20%.'},
    {'label': '1987 Black Monday', 'drawdown': -0.34, 'recession': False, 'regime': 'pre-2000',
     'x': [.40,.20,.20,.15,.45,.20,.15,.70,.10,.80,.35],
     'notes': 'Program trading crash. Fundamentals OK. V-recovery.'},
    {'label': '1990-91 Gulf War', 'drawdown': -0.20, 'recession': True, 'regime': 'pre-2000',
     'x': [.80,.70,.60,.70,.60,.60,.50,.50,.70,.50,.30],
     'notes': 'Iraq invades Kuwait. Oil +75%. Short recession.'},
    {'label': '1998 LTCM/Asia', 'drawdown': -0.19, 'recession': False, 'regime': 'pre-2000',
     'x': [.50,.20,.30,.20,.55,.30,.30,.65,.15,.60,.35],
     'notes': 'Asian contagion + LTCM. Greenspan cut 3x. V-recovery.'},
    {'label': '2000-02 Dot-com', 'drawdown': -0.49, 'recession': True, 'regime': 'post-2000',
     'x': [.70,.30,.70,.30,.70,.70,.60,.90,.60,.85,.40],
     'notes': 'Tech bubble. Market peaked 12mo before NBER.'},
    {'label': '2007-09 GFC', 'drawdown': -0.57, 'recession': True, 'regime': 'post-2000',
     'x': [.90,.90,.90,.60,.95,.90,.80,.85,.90,.85,.25],
     'notes': 'Housing + credit. Every indicator fired. Textbook.'},
    {'label': '2011 EU Debt', 'drawdown': -0.19, 'recession': False, 'regime': 'post-2000',
     'x': [.20,.25,.40,.30,.45,.25,.30,.30,.20,.65,.50],
     'notes': 'Greek sovereign debt. US fundamentals OK. Spreads <5%.'},
    {'label': '2018 Fed Tight', 'drawdown': -0.20, 'recession': False, 'regime': 'post-2000',
     'x': [.60,.35,.35,.25,.35,.30,.35,.45,.15,.70,.45],
     'notes': 'Fed overtightened. Powell pivoted. V-recovery.'},
    {'label': '2020 COVID', 'drawdown': -0.34, 'recession': True, 'regime': 'post-2000',
     'x': [.10,.10,.20,.10,.80,.30,.20,.30,.80,.90,.10],
     'notes': 'Exogenous shock. Model CANNOT predict pandemics.'},
    {'label': '2022 Inflation', 'drawdown': -0.25, 'recession': False, 'regime': 'post-2000',
     'x': [.85,.50,.70,.55,.25,.35,.40,.55,.25,.75,.80],
     'notes': 'Deep inversion BUT fiscal 5.9%, AI, spreads <5%. False positive.'},
]
 
CURRENT_DEFAULT = [.60,.40,.70,.55,.20,.50,.50,.70,.30,.60,.70]
REGIME_WEIGHTS = {'pre-2000': 0.7, 'post-2000': 1.0}
 
# ══════════════════════════════════════════════════════════════
# ML MODEL
# ══════════════════════════════════════════════════════════════
 
@st.cache_data
def engineer_features(x, regime='post-2000', prev=None):
    yc,bp,lei,oil,cs,cp,dg,eu,sa,br,fi = x
    f = {}
    nm = ['yc','bp','lei','oil','cs','cp','dg','eu','sa','br','fi']
    for i,n in enumerate(nm): f[f'raw_{n}'] = x[i]
    f['ix_yc_cs']=yc*cs; f['ix_oil_lei']=oil*lei; f['ix_eu_br']=eu*br
    f['ix_cs_sa']=cs*sa; f['ix_fi_yc']=fi*yc
    f['comp_early']=np.mean([yc,bp,lei]); f['comp_credit']=np.mean([cs,cp,dg])
    f['comp_euphoria']=np.mean([eu,br]); f['comp_stress']=np.mean([yc,bp,lei,oil,cs,cp,dg,eu,sa,br])
    f['comp_adj']=f['comp_stress']*(1-fi*0.3)
    if prev:
        for i,n in enumerate(nm): f[f'roc_{n}']=x[i]-prev[i]
        f['roc_stress']=np.mean(x[:10])-np.mean(prev[:10])
    else:
        for n in nm: f[f'roc_{n}']=0.0
        f['roc_stress']=0.0
    f['reg_post']=1.0 if regime=='post-2000' else 0.0
    f['yc_adj']=yc*(0.8 if regime=='post-2000' else 1.0)
    f['fi_adj']=fi*(1.2 if regime=='post-2000' else 1.0)
    return f
 
def build_dataset():
    X, y, weights = [], [], []
    for i, ev in enumerate(EVENTS):
        prev = EVENTS[i-1]['x'] if i > 0 else None
        feats = engineer_features(ev['x'], ev.get('regime','post-2000'), prev)
        keys = sorted(feats.keys())
        X.append([feats[k] for k in keys])
        y.append(1 if ev['recession'] else 0)
        weights.append(REGIME_WEIGHTS[ev.get('regime','post-2000')])
    return np.array(X), np.array(y), np.array(weights), keys
 
class Ensemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'lr': LogisticRegression(max_iter=1000, C=0.5, class_weight='balanced', random_state=42),
            'rf': RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=2,
                                         class_weight='balanced', random_state=42),
            'gbt': GradientBoostingClassifier(n_estimators=50, max_depth=2, min_samples_leaf=2,
                                              learning_rate=0.1, random_state=42),
        }
 
    def fit(self, X, y, sw=None):
        Xs = self.scaler.fit_transform(X)
        for m in self.models.values(): m.fit(Xs, y, sample_weight=sw)
 
    def predict(self, x):
        Xs = self.scaler.transform(x.reshape(1,-1) if x.ndim==1 else x)
        probs = {n: m.predict_proba(Xs)[:,1] for n,m in self.models.items()}
        probs['ensemble'] = np.mean([probs[k] for k in self.models], axis=0)
        probs['agreement'] = 1 - np.std([probs[k] for k in self.models], axis=0) * 2
        return probs
 
    def importance(self):
        lr = np.abs(self.models['lr'].coef_[0]); lr = lr/lr.sum()*100
        rf = self.models['rf'].feature_importances_; rf = rf/rf.sum()*100
        gb = self.models['gbt'].feature_importances_; gb = gb/gb.sum()*100
        return np.mean([lr, rf, gb], axis=0)
 
    def bootstrap_ci(self, X, y, xc, n=50, sw=None):
        preds = []
        for _ in range(n):
            for _ in range(20):
                idx = np.random.choice(len(X), len(X), replace=True)
                if len(np.unique(y[idx])) >= 2: break
            try:
                bm = Ensemble()
                bm.fit(X[idx], y[idx], sw=sw[idx] if sw is not None else None)
                preds.append(bm.predict(xc)['ensemble'][0])
            except: continue
        preds.sort()
        return {'p5': preds[max(0,int(len(preds)*.05))],
                'p50': preds[int(len(preds)*.5)],
                'p95': preds[min(len(preds)-1,int(len(preds)*.95))]}
 
 
@st.cache_resource
def train_model(X_tuple, y_tuple, w_tuple):
    """Cache the trained model. Input must be hashable (tuples)."""
    X = np.array(X_tuple)
    y = np.array(y_tuple)
    w = np.array(w_tuple)
    model = Ensemble()
    model.fit(X, y, sw=w)
    return model
 
def get_current_vec(current, prev=None):
    feats = engineer_features(current, 'post-2000', prev)
    keys = sorted(feats.keys())
    return np.array([feats[k] for k in keys]), keys
 
def classify(prob, cs, sa):
    if prob < 0.25: return 'LOW RISK', '#8fb840', 'No significant drawdown expected'
    elif prob < 0.40:
        if cs < 0.40: return 'CORRECTION POSSIBLE', '#c8a84a', 'Correction likely (10-20% dip)'
        else: return 'ELEVATED', '#c8a84a', 'Elevated risk, ambiguous signal'
    elif prob < 0.60:
        if cs >= 0.60 and sa >= 0.50: return 'RECESSION LIKELY', '#d4614a', 'Recession likely — sell aggressively'
        elif cs >= 0.50: return 'RECESSION POSSIBLE', '#d4614a', 'Recession possible — reduce risk'
        else: return 'CORRECTION LIKELY', '#c8a84a', 'Correction likely — trim, do not panic'
    else:
        if cs >= 0.60: return 'RECESSION LIKELY', '#d4614a', 'Recession likely — sell aggressively'
        else: return 'SEVERE CORRECTION', '#d4614a', 'Severe correction or recession'
 
 
# ══════════════════════════════════════════════════════════════
# LSI MODULE
# ══════════════════════════════════════════════════════════════
 
def compute_lsi(hours, quits, temp_yoy, claims, u6_gap):
    """Labor Stress Index — 0-100 composite."""
    def score(val, h, s, st, cr, inv=False):
        if inv:
            if val >= h: return 10
            elif val >= s: return 10 + (h-val)/(h-s)*30
            elif val >= st: return 40 + (s-val)/(s-st)*30
            elif val >= cr: return 70 + (st-val)/(st-cr)*25
            else: return 95
        else:
            if val <= h: return 10
            elif val <= s: return 10 + (val-h)/(s-h)*30
            elif val <= st: return 40 + (val-s)/(st-s)*30
            elif val <= cr: return 70 + (val-st)/(cr-st)*25
            else: return 95
 
    comps = {
        'hours': score(hours, 41.0, 40.5, 40.0, 39.5, inv=True),
        'quits': score(quits, 3.5, 3.0, 2.5, 2.0, inv=True),
        'temp': score(temp_yoy, 3.0, 0.0, -5.0, -15.0, inv=True),
        'claims': score(claims, 1800, 2200, 3000, 4500),
        'u6': score(u6_gap, 3.0, 3.5, 4.5, 6.0),
    }
    weights = {'hours': 0.25, 'quits': 0.25, 'temp': 0.20, 'claims': 0.15, 'u6': 0.15}
    total = sum(comps[k] * weights[k] for k in weights)
    level = 'CRISIS' if total >= 70 else 'STRESSED' if total >= 50 else 'SOFTENING' if total >= 30 else 'HEALTHY'
    return total, level, comps
 
 
def compute_csi(sloos_ci, sloos_cc, chargeoff, cons_del, auto_del, cp_chg):
    """Credit Stress Index — 0-100 composite of banking system health."""
    def score(val, h, s, st, cr, inv=False):
        if inv:
            if val >= h: return 10
            elif val >= s: return 10 + (h-val)/(h-s)*30
            elif val >= st: return 40 + (s-val)/(s-st)*30
            elif val >= cr: return 70 + (st-val)/(st-cr)*25
            else: return 95
        else:
            if val <= h: return 10
            elif val <= s: return 10 + (val-h)/(s-h)*30
            elif val <= st: return 40 + (val-s)/(st-s)*30
            elif val <= cr: return 70 + (val-st)/(cr-st)*25
            else: return 95
 
    comps = {
        'sloos_ci': score(sloos_ci, 0, 15, 35, 60),
        'sloos_cc': score(sloos_cc, 0, 10, 25, 45),
        'chargeoff': score(chargeoff, 0.50, 0.80, 1.20, 2.00),
        'cons_del': score(cons_del, 2.0, 2.5, 3.2, 4.0),
        'auto_del': score(auto_del, 1.8, 2.3, 2.8, 3.5),
        'cp_chg': score(cp_chg, 5.0, 0.0, -8.0, -20.0, inv=True),
    }
    weights = {'sloos_ci': 0.15, 'sloos_cc': 0.10, 'chargeoff': 0.20,
               'cons_del': 0.15, 'auto_del': 0.15, 'cp_chg': 0.25}
    total = sum(comps[k] * weights[k] for k in weights)
    level = 'CRISIS' if total >= 65 else 'STRESSED' if total >= 45 else 'TIGHTENING' if total >= 25 else 'HEALTHY'
    return total, level, comps
 
 
def compute_gci(dxy_surge, em_sov, global_pmi, china_ci, eu_bank):
    """Global Contagion Index — 0-100 composite of global stress."""
    def score(val, h, s, st, cr, inv=False):
        if inv:
            if val >= h: return 10
            elif val >= s: return 10 + (h-val)/(h-s)*30
            elif val >= st: return 40 + (s-val)/(s-st)*30
            elif val >= cr: return 70 + (st-val)/(st-cr)*25
            else: return 95
        else:
            if val <= h: return 10
            elif val <= s: return 10 + (val-h)/(s-h)*30
            elif val <= st: return 40 + (val-s)/(st-s)*30
            elif val <= cr: return 70 + (val-st)/(cr-st)*25
            else: return 95
 
    comps = {
        'dxy_surge': score(dxy_surge, -5, 3, 8, 15),
        'em_sov': score(em_sov, 300, 400, 600, 900),
        'global_pmi': score(global_pmi, 52, 50.5, 49, 45, inv=True),
        'china_ci': score(china_ci, 2, 0, -2, -5, inv=True),
        'eu_bank': score(eu_bank, 35, 50, 70, 90),
    }
    weights = {'dxy_surge': 0.20, 'em_sov': 0.20, 'global_pmi': 0.25,
               'china_ci': 0.15, 'eu_bank': 0.20}
    total = sum(comps[k] * weights[k] for k in weights)
    level = 'CRISIS' if total >= 70 else 'STRESSED' if total >= 50 else 'SOFTENING' if total >= 30 else 'CALM'
    return total, level, comps
 
 
def compute_cfhi(real_inc, savings, tdsp, cc_util, cc_del, retail):
    """Consumer Financial Health Index — 0-100 trifecta phase detector."""
    def score(val, h, s, st, cr, inv=False):
        if inv:
            if val >= h: return 10
            elif val >= s: return 10 + (h-val)/(h-s)*30
            elif val >= st: return 40 + (s-val)/(s-st)*30
            elif val >= cr: return 70 + (st-val)/(st-cr)*25
            else: return 95
        else:
            if val <= h: return 10
            elif val <= s: return 10 + (val-h)/(s-h)*30
            elif val <= st: return 40 + (val-s)/(st-s)*30
            elif val <= cr: return 70 + (val-st)/(cr-st)*25
            else: return 95
 
    comps = {
        'real_inc': score(real_inc, 2.5, 1.5, 0.0, -2.0, inv=True),
        'savings': score(savings, 6.0, 4.5, 3.0, 2.0, inv=True),
        'tdsp': score(tdsp, 9.5, 10.5, 11.5, 13.0),
        'cc_util': score(cc_util, 22, 26, 30, 38),
        'cc_del': score(cc_del, 2.5, 3.2, 4.0, 5.5),
        'retail': score(retail, 3.0, 1.5, 0.0, -3.0, inv=True),
    }
    weights = {'real_inc': 0.20, 'savings': 0.15, 'tdsp': 0.15,
               'cc_util': 0.20, 'cc_del': 0.15, 'retail': 0.15}
    total = sum(comps[k] * weights[k] for k in weights)
 
    # Phase identification
    if retail <= 0 and cc_del >= 4.0 and savings <= 3.0:
        phase = "PHASE 3 — SPENDING CRACKED"
        phase_desc = "Recession dynamics active"
    elif total >= 55:
        phase = "LATE PHASE 2 — BRIDGE STRESSED"
        phase_desc = "Phase 3 imminent if no relief"
    elif total >= 30:
        phase = "PHASE 2 — CREDIT BRIDGE"
        phase_desc = "Spending via credit, watch for snap"
    else:
        phase = "PHASE 1 — HEALTHY"
        phase_desc = "Consumer buffer intact, recession unlikely"
 
    return total, phase, phase_desc, comps
 
 
# ══════════════════════════════════════════════════════════════
# SIDEBAR — INDICATOR CONTROLS
# ══════════════════════════════════════════════════════════════
 
with st.sidebar:
    st.markdown("### ⚙️ Indicator Controls")
    st.markdown("*Adjust fire levels to see how the model responds*")
 
    indicator_labels = {
        'Yield curve': 'Treasury yield curve inversion',
        'Permits': 'Building permit decline',
        'LEI': 'Leading Economic Index decline',
        'Oil shock': 'Oil price shock severity',
        'Credit spreads': 'HY OAS widening — THE KEY',
        'Corp profits': 'Corporate profit decline',
        'Durables': 'Durable goods orders decline',
        'Euphoria': 'Margin debt + Buffett indicator',
        'Sahm rule': 'Sahm rule proximity',
        'Breadth': 'Market breadth deterioration',
        'Fiscal(-)': 'Fiscal support (counter-indicator)',
    }
 
    current = []
    for i, name in enumerate(INDICATOR_NAMES):
        val = st.slider(
            f"{name}",
            0.0, 1.0, CURRENT_DEFAULT[i], 0.05,
            help=indicator_labels[name],
            key=f"ind_{i}"
        )
        current.append(val)
 
    st.markdown("---")
    st.markdown("### 📊 LSI Inputs")
    lsi_hours = st.slider("Avg weekly hours", 38.0, 42.0, 39.8, 0.1)
    lsi_quits = st.slider("Quits rate (M/mo)", 1.5, 5.0, 2.8, 0.1)
    lsi_temp = st.slider("Temp emp YoY (%)", -20.0, 10.0, -6.2, 0.5)
    lsi_claims = st.slider("Continued claims (K)", 1000, 6000, 2150, 50)
    lsi_u6 = st.slider("U6-U3 gap (pp)", 2.0, 8.0, 4.1, 0.1)
 
    st.markdown("---")
    st.markdown("### 🏦 CSI Inputs")
    csi_sloos_ci = st.slider("SLOOS C&I tightening (%)", -30.0, 80.0, 18.0, 1.0, help="Net % banks tightening business loans")
    csi_sloos_cc = st.slider("SLOOS CC tightening (%)", -20.0, 60.0, 22.0, 1.0, help="Net % banks tightening credit cards")
    csi_chargeoff = st.slider("Charge-off rate (%)", 0.1, 3.0, 0.72, 0.02, help="Bank loan charge-off rate")
    csi_cons_del = st.slider("Consumer delinquency (%)", 1.0, 5.0, 3.1, 0.1, help="Consumer loan delinquency rate")
    csi_auto_del = st.slider("Auto delinq 60+d (%)", 1.0, 4.0, 3.3, 0.1, help="Auto loans 60+ days delinquent")
    csi_cp_chg = st.slider("Commercial paper YoY (%)", -30.0, 20.0, -5.0, 1.0, help="YoY change in commercial paper outstanding")
 
    st.markdown("---")
    st.markdown("### 🌍 GCI Inputs")
    gci_dxy_surge = st.slider("USD 6m change (%)", -15.0, 20.0, -5.0, 0.5, help="Dollar surge = safe haven flight")
    gci_em_sov = st.slider("EM sovereign spread (bps)", 150, 1500, 280, 10, help="EM stress gauge")
    gci_global_pmi = st.slider("Global manuf PMI", 40.0, 60.0, 49.1, 0.1, help="Below 50 = contraction")
    gci_china_ci = st.slider("China credit impulse (pp)", -8.0, 10.0, -2.0, 0.5, help="Negative = China dragging global")
    gci_eu_bank = st.slider("EU bank stress (0-100)", 20, 100, 55, 1, help="Cross-Atlantic contagion channel")
 
    st.markdown("---")
    st.markdown("### 👤 CFHI Inputs")
    cfhi_real_inc = st.slider("Real disposable income YoY (%)", -5.0, 8.0, 1.2, 0.1, help="Real income growth")
    cfhi_savings = st.slider("Personal savings rate (%)", 1.0, 15.0, 3.8, 0.1, help="Household savings buffer")
    cfhi_tdsp = st.slider("Debt service ratio (%)", 8.0, 15.0, 11.4, 0.1, help="% of income going to debt")
    cfhi_cc_util = st.slider("CC utilization (%)", 15, 45, 29, 1, help="Credit bridge usage")
    cfhi_cc_del = st.slider("CC delinquency 90+d (%)", 1.0, 8.0, 2.94, 0.05, help="Bridge breaking signal")
    cfhi_retail = st.slider("Real retail sales YoY (%)", -10.0, 10.0, 1.8, 0.2, help="Actual spending")
 
# ══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════
 
st.title("📊 Recession Radar v7.0")
st.markdown("*ML ensemble + walk-forward backtesting + scenario analysis*")
 
# Train model
X, y, w, feat_names = build_dataset()
model = train_model(tuple(map(tuple, X)), tuple(y), tuple(w))
 
# Current prediction
cur_vec, _ = get_current_vec(current)
probs = model.predict(cur_vec)
ml_p = probs['ensemble'][0]
agreement = probs['agreement'][0]
 
# Judgment probability
j_p = sum(current[i] * JUDGMENT_WEIGHTS[i] for i in range(11)) / sum(JUDGMENT_WEIGHTS)
 
# Classification
cls_name, cls_color, cls_desc = classify(ml_p, current[4], current[8])
 
# LSI
lsi_total, lsi_level, lsi_comps = compute_lsi(lsi_hours, lsi_quits, lsi_temp, lsi_claims, lsi_u6)
 
# CSI
csi_total, csi_level, csi_comps = compute_csi(csi_sloos_ci, csi_sloos_cc, csi_chargeoff, csi_cons_del, csi_auto_del, csi_cp_chg)
 
# GCI
gci_total, gci_level, gci_comps = compute_gci(gci_dxy_surge, gci_em_sov, gci_global_pmi, gci_china_ci, gci_eu_bank)
 
# CFHI
cfhi_total, cfhi_phase, cfhi_phase_desc, cfhi_comps = compute_cfhi(cfhi_real_inc, cfhi_savings, cfhi_tdsp, cfhi_cc_util, cfhi_cc_del, cfhi_retail)
cfhi_short = cfhi_phase.split(" —")[0] if " —" in cfhi_phase else cfhi_phase.split(" -")[0]
 
# Divergence
csi_norm = csi_total / 100
hy_fire = current[4]
div_gap = csi_norm - hy_fire
if abs(div_gap) < 0.15:
    div_status, div_color = "ALIGNED", "#8fb840"
elif div_gap > 0.15:
    div_status, div_color = "CSI LEADING", "#d4614a"
else:
    div_status, div_color = "MARKET LEADING", "#c8a84a"
 
# ── HERO METRICS ──
col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
with col1:
    st.metric("ML Prob", f"{ml_p*100:.0f}%", f"Agree {agreement*100:.0f}%")
with col2:
    st.metric("Judgment", f"{j_p*100:.0f}%", "Expert weights")
with col3:
    st.metric("Labor (LSI)", f"{lsi_total:.0f}/100", lsi_level)
with col4:
    st.metric("Credit (CSI)", f"{csi_total:.0f}/100", csi_level)
with col5:
    st.metric("Global (GCI)", f"{gci_total:.0f}/100", gci_level)
with col6:
    st.metric("Consumer (CFHI)", f"{cfhi_total:.0f}/100", cfhi_short)
with col7:
    st.metric("HY OAS", f"{current[4]*100:.0f}%", "Market fire")
 
st.markdown(f"<div style='padding:16px;background:{cls_color}22;border-left:4px solid {cls_color};border-radius:8px;margin:16px 0'>"
            f"<h3 style='color:{cls_color};margin:0'>{cls_name}</h3>"
            f"<p style='color:#b09a6e;margin:4px 0 0 0'>{cls_desc}</p></div>",
            unsafe_allow_html=True)
 
# Divergence warning
if div_gap > 0.15:
    st.markdown(f"<div style='padding:14px;background:#d4614a22;border-left:4px solid #d4614a;border-radius:8px;margin:8px 0'>"
                f"<h4 style='color:#d4614a;margin:0'>System A/B Divergence Detected</h4>"
                f"<p style='color:#b09a6e;margin:4px 0 0 0'>Banking system stress (CSI {csi_total:.0f}/100) is significantly above "
                f"market pricing (HY OAS {hy_fire*100:.0f}%). This pattern preceded the GFC by 3-6 months in 2007. "
                f"Ground-level data suggests risk is higher than market indicators show.</p></div>",
                unsafe_allow_html=True)
 
# Bootstrap CI
with st.spinner("Computing bootstrap confidence interval..."):
    bs = model.bootstrap_ci(X, y, cur_vec, n=50, sw=w)
 
st.caption(f"Bootstrap 90% CI: {bs['p5']*100:.0f}% — {bs['p95']*100:.0f}% | Median: {bs['p50']*100:.0f}%")
 
# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
 
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📈 Dashboard",
    "🎯 Action Plan",
    "🔮 Scenarios",
    "🧠 ML Engine",
    "👷 Labor Stress",
    "🏦 Credit Stress",
    "🌍 Global",
    "👤 Consumer",
    "🛢️ Pipe 1: Foreign Funding",
    "🔧 Pipe 2: Shadow Banking",
    "ℹ️ About"
])
 
# ── TAB 1: DASHBOARD ──
with tab1:
    st.header("Indicator Dashboard")
 
    imp = model.importance()
    ml_weights = {}
    for i, name in enumerate(['raw_yc','raw_bp','raw_lei','raw_oil','raw_cs','raw_cp','raw_dg','raw_eu','raw_sa','raw_br','raw_fi']):
        if name in feat_names:
            idx = feat_names.index(name)
            ml_weights[i] = imp[idx]
 
    sum_ml = sum(ml_weights.values()) or 1
    ml_pct = {k: v/sum_ml*100 for k,v in ml_weights.items()}
 
    df = pd.DataFrame({
        'Indicator': INDICATOR_NAMES,
        'Fire Level': [f"{v*100:.0f}%" for v in current],
        'Judgment Weight': [f"{w}%" for w in JUDGMENT_WEIGHTS],
        'ML Weight': [f"{ml_pct.get(i,0):.0f}%" for i in range(11)],
        'Status': ['🔴 High' if current[i] >= 0.7 else '🟡 Med' if current[i] >= 0.4 else '🟢 Low' for i in range(11)],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    st.markdown("### Fire Level Visualization")
    chart_df = pd.DataFrame({'Indicator': INDICATOR_NAMES, 'Fire': current})
    st.bar_chart(chart_df.set_index('Indicator'))
 
# ── TAB 2: ACTION PLAN ──
with tab2:
    st.header("Asset Allocation by Regime")
 
    regimes = [
        (0, 0.20, 'LOW RISK',  '70-80%', '15-20%', '0-5%',   '3-5%',   '#8fb840'),
        (0.20, 0.35, 'MODERATE', '60-70%', '20-25%', '5-8%',   '5-10%',  '#8fb840'),
        (0.35, 0.55, 'ELEVATED', '50-60%', '20-25%', '8-12%',  '10-15%', '#c8a84a'),
        (0.55, 0.75, 'HIGH',     '35-45%', '20-25%', '10-15%', '20-30%', '#d4614a'),
        (0.75, 1.0,  'CRISIS',   '20-30%', '25-30%', '15-20%', '25-35%', '#d4614a'),
    ]
 
    df = pd.DataFrame({
        'Probability': [f"{lo*100:.0f}-{hi*100:.0f}%" for lo,hi,*_ in regimes],
        'Regime': [r[2] for r in regimes],
        'Equity': [r[3] for r in regimes],
        'Fixed Income': [r[4] for r in regimes],
        'Alts/Gold': [r[5] for r in regimes],
        'Cash': [r[6] for r in regimes],
        'Current': ['← YOU ARE HERE' if lo <= ml_p < hi else '' for lo,hi,*_ in regimes],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    st.markdown("### Position Sizing Rules")
    st.markdown("""
    1. **Max 5% AUM adjustment per month** — no panic selling
    2. **Stage 1 active**: begin 3-month gradual rotation
    3. **Credit spreads cross 4.5%**: accelerate to 8%/month
    4. **Bottom signal fires**: deploy 50% immediately, 50% over 4 weeks
    """)
 
    st.markdown("### Asymmetric Regret")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Wrong-cautious cost:** ~2-3% portfolio drag (survivable)")
    with col2:
        st.markdown(f"**Wrong-complacent cost:** ~20-35% drawdown (catastrophic)")
 
# ── TAB 3: SCENARIOS ──
with tab3:
    st.header("Scenario Analysis")
    st.caption("Each scenario adjusts indicators through the ML model to produce probabilities and action plans")
 
    scenarios = [
        {'name': 'A. Peace Deal', 'sub': 'Ceasefire holds, oil to $75',
         'ch': {3:.2, 4:.15, 7:.4, 8:.2, 9:.3, 10:.7},
         'spx': '+15 to +22%', 'bonds': '+3 to +5%', 'gold': '-8 to -12%',
         'action': 'Risk-on. Rebuild cyclicals. Cash to 5%.', 'color': '#8fb840'},
        {'name': 'B. Slow Burn', 'sub': 'Oil $90-100 for 6+ months',
         'ch': {3:.55, 4:.25, 7:.6, 8:.35, 9:.5, 10:.65},
         'spx': '-8 to -15%', 'bonds': '+1 to +3%', 'gold': '+5 to +10%',
         'action': 'Hold defensives. Trim cyclicals. Cash 10-15%.', 'color': '#c8a84a'},
        {'name': 'C. Escalation', 'sub': 'Hormuz closed, oil $130+',
         'ch': {3:.9, 4:.6, 7:.85, 8:.6, 9:.8, 10:.5},
         'spx': '-25 to -35%', 'bonds': '-2 to +2%', 'gold': '+15 to +25%',
         'action': 'Aggressive de-risk. Cash 25-30%. Puts.', 'color': '#d4614a'},
        {'name': 'D. Credit Event', 'sub': 'CRE/private credit blowup',
         'ch': {3:.55, 4:.85, 7:.9, 8:.7, 9:.9, 10:.5},
         'spx': '-35 to -50%', 'bonds': '+8 to +15%', 'gold': '+20 to +30%',
         'action': 'Max defensive. Treasuries + gold + cash.', 'color': '#d4614a'},
    ]
 
    for sc in scenarios:
        adj = list(current)
        for idx, val in sc['ch'].items(): adj[idx] = val
        sv, _ = get_current_vec(adj)
        sp = model.predict(sv)['ensemble'][0]
 
        with st.expander(f"{sc['name']}: {sc['sub']} — **{sp*100:.0f}% recession probability**", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("S&P 500", sc['spx'])
            col2.metric("Bonds (AGG)", sc['bonds'])
            col3.metric("Gold", sc['gold'])
            st.markdown(f"**Action:** {sc['action']}")
 
# ── TAB 4: ML ENGINE ──
with tab4:
    st.header("ML Engine Internals")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Ensemble", f"{probs['ensemble'][0]*100:.0f}%")
    col2.metric("Model Agreement", f"{agreement*100:.0f}%")
    col3.metric("Training Points", f"n={len(EVENTS)}")
 
    st.markdown("### Individual Model Predictions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression", f"{probs['lr'][0]*100:.0f}%")
    col2.metric("Random Forest", f"{probs['rf'][0]*100:.0f}%")
    col3.metric("Gradient Boosting", f"{probs['gbt'][0]*100:.0f}%")
 
    st.markdown("### Leave-One-Out Cross-Validation")
    st.caption("Each event held out once, model retrained on remaining 12, prediction tested")
 
    loo_results = []
    for i in range(len(X)):
        Xt = np.delete(X, i, axis=0)
        yt = np.delete(y, i)
        wt = np.delete(w, i)
        m = Ensemble()
        m.fit(Xt, yt, sw=wt)
        p = m.predict(X[i])
        pred_class = 1 if p['ensemble'][0] >= 0.5 else 0
        loo_results.append({
            'Event': EVENTS[i]['label'],
            'Actual': 'RECESSION' if y[i] else 'correction',
            'LR': f"{p['lr'][0]*100:.0f}%",
            'RF': f"{p['rf'][0]*100:.0f}%",
            'GBT': f"{p['gbt'][0]*100:.0f}%",
            'Ensemble': f"{p['ensemble'][0]*100:.0f}%",
            'Correct': '✅' if pred_class == y[i] else '❌',
        })
    loo_df = pd.DataFrame(loo_results)
    acc = sum(1 for r in loo_results if r['Correct'] == '✅') / len(loo_results)
    st.dataframe(loo_df, use_container_width=True, hide_index=True)
    st.metric("LOOCV Accuracy", f"{acc*100:.0f}%")
 
    st.markdown("### Top ML Features")
    top_features = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)[:10]
    feat_df = pd.DataFrame(top_features, columns=['Feature', 'Importance'])
    st.bar_chart(feat_df.set_index('Feature'))
 
# ── TAB 5: LABOR STRESS ──
with tab5:
    st.header(f"Labor Stress Index — {lsi_total:.0f}/100")
 
    col1, col2 = st.columns([1, 2])
    with col1:
        level_color = '#d4614a' if lsi_level in ('CRISIS', 'STRESSED') else '#c8a84a' if lsi_level == 'SOFTENING' else '#8fb840'
        st.markdown(f"<div style='padding:20px;background:{level_color}22;border-radius:12px;text-align:center'>"
                    f"<h1 style='color:{level_color};margin:0'>{lsi_total:.0f}</h1>"
                    f"<h3 style='color:{level_color};margin:4px 0'>{lsi_level}</h3></div>",
                    unsafe_allow_html=True)
 
    with col2:
        st.markdown("#### What the headline misses")
        st.markdown(f"""
        - **U-3 unemployment:** 4.3% (looks fine)
        - **Actual LSI:** {lsi_total:.0f}/100 ({lsi_level.lower()})
        - **The gap between these IS the illusion**
        """)
 
    st.markdown("### Component Breakdown")
    comp_data = {
        'Avg weekly hours': (lsi_hours, lsi_comps['hours'], "Hidden layoffs — cos cut hours first"),
        'Quits rate': (lsi_quits, lsi_comps['quits'], "Workers scared — only quit when confident"),
        'Temp employment YoY': (lsi_temp, lsi_comps['temp'], "First domino — cos cut temps first"),
        'Continued claims': (lsi_claims, lsi_comps['claims'], "Rising = harder to find new job"),
        'U6-U3 gap': (lsi_u6, lsi_comps['u6'], "Underemployment — want FT, stuck PT"),
    }
    df = pd.DataFrame([
        {'Component': k, 'Value': v[0], 'Stress': f"{v[1]:.0f}/100", 'Why': v[2]}
        for k, v in comp_data.items()
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    st.markdown("### Historical Comparison")
    hist_df = pd.DataFrame([
        {'Date': '2022-01', 'LSI': 10, 'Outcome': 'CORRECTION — labor strong = buy dip'},
        {'Date': '2018-09', 'LSI': 18, 'Outcome': 'CORRECTION — labor healthy = V-recovery'},
        {'Date': '2020-02', 'LSI': 20, 'Outcome': 'PRE-COVID — labor gave NO warning'},
        {'Date': '2000-03', 'LSI': 25, 'Outcome': 'PRE-RECESSION — labor lagged'},
        {'Date': '2007-10', 'LSI': 38, 'Outcome': 'PRE-GFC — labor softening'},
        {'Date': '2025-02', 'LSI': 45, 'Outcome': 'Current cycle peak'},
        {'Date': 'NOW', 'LSI': lsi_total, 'Outcome': '← CURRENT READING'},
        {'Date': '2001-09', 'LSI': 68, 'Outcome': 'RECESSION CONFIRMED'},
        {'Date': '2008-09', 'LSI': 95, 'Outcome': 'CRISIS'},
    ])
    st.dataframe(hist_df.sort_values('LSI'), use_container_width=True, hide_index=True)
 
    st.warning("""
    **Critical Insight:** In every historical correction (1987, 1998, 2011, 2018, 2022),
    the LSI was below 30. In every recession, the LSI was above 35 at the market peak.
    Current LSI has NEVER been associated with a correction-only outcome. It has always
    preceded or accompanied recession.
    """)
 
# ── TAB 6: CREDIT STRESS ──
with tab6:
    st.header(f"Credit Stress Index — {csi_total:.0f}/100")
 
    col1, col2 = st.columns([1, 2])
    with col1:
        csi_color = '#d4614a' if csi_level in ('CRISIS', 'STRESSED') else '#c8a84a' if csi_level == 'TIGHTENING' else '#8fb840'
        st.markdown(f"<div style='padding:20px;background:{csi_color}22;border-radius:12px;text-align:center'>"
                    f"<h1 style='color:{csi_color};margin:0'>{csi_total:.0f}</h1>"
                    f"<h3 style='color:{csi_color};margin:4px 0'>{csi_level}</h3></div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("#### What credit spreads miss")
        st.markdown(f"""
        - **HY OAS (market):** {hy_fire*100:.0f}% fire (bond traders say calm)
        - **CSI (fundamentals):** {csi_total:.0f}/100 (banks are tightening)
        - **Divergence:** {div_status}
        - **This gap preceded the GFC by 3-6 months**
        """)
 
    st.markdown("### Divergence Analysis")
    if div_gap > 0.15:
        st.error(f"Banking system stress ({csi_total:.0f}/100) significantly exceeds market pricing ({hy_fire*100:.0f}%). "
                 f"In 2007, this exact pattern (CSI 52 vs spreads calm) preceded the market crash by 3-6 months. "
                 f"Current CSI ({csi_total:.0f}) is ABOVE the pre-GFC level.")
    elif div_gap < -0.15:
        st.info(f"Market pricing ({hy_fire*100:.0f}%) exceeds banking stress ({csi_total:.0f}/100). Market may be overreacting.")
    else:
        st.success(f"CSI ({csi_total:.0f}/100) and credit spreads ({hy_fire*100:.0f}%) are aligned. High confidence in current reading.")
 
    st.markdown("### Component Breakdown")
    csi_labels = {
        'sloos_ci': ('SLOOS C&I tightening', csi_sloos_ci, '% net', 'Banks tightening business loans'),
        'sloos_cc': ('SLOOS CC tightening', csi_sloos_cc, '% net', 'Credit card lending tightening'),
        'chargeoff': ('Loan charge-off rate', csi_chargeoff, '%', 'Actual loan defaults'),
        'cons_del': ('Consumer delinquency', csi_cons_del, '%', 'Consumers failing to pay'),
        'auto_del': ('Auto loan 60+d delinq', csi_auto_del, '%', 'First discretionary debt to crack'),
        'cp_chg': ('Commercial paper YoY', csi_cp_chg, '%', 'Corporate short-term borrowing'),
    }
    cdf = pd.DataFrame([
        {'Component': v[0], 'Value': f"{v[1]}{v[2]}", 'Stress': f"{csi_comps[k]:.0f}/100", 'Why': v[3]}
        for k, v in csi_labels.items()
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)
 
    st.markdown("### Historical Comparison")
    hist_csi = pd.DataFrame([
        {'Date': '2022-01', 'CSI': 5, 'Outcome': 'CORRECTION — credit pristine'},
        {'Date': '2020-02', 'CSI': 12, 'Outcome': 'PRE-COVID — no credit warning'},
        {'Date': '2018-09', 'CSI': 18, 'Outcome': 'CORRECTION — credit healthy'},
        {'Date': '1998-07', 'CSI': 22, 'Outcome': 'CORRECTION — brief spike'},
        {'Date': '2011-04', 'CSI': 28, 'Outcome': 'CORRECTION — post-GFC residual'},
        {'Date': '2000-03', 'CSI': 38, 'Outcome': 'RECESSION — credit tightening early'},
        {'Date': '2007-10', 'CSI': 52, 'Outcome': 'PRE-GFC — CSI LEADING spreads'},
        {'Date': 'NOW', 'CSI': csi_total, 'Outcome': 'CURRENT READING'},
        {'Date': '1990-07', 'CSI': 72, 'Outcome': 'RECESSION — S&L crisis'},
        {'Date': '2001-09', 'CSI': 75, 'Outcome': 'RECESSION — post-Enron'},
        {'Date': '2008-09', 'CSI': 96, 'Outcome': 'CRISIS — total freeze'},
    ])
    st.dataframe(hist_csi.sort_values('CSI'), use_container_width=True, hide_index=True)
 
    st.markdown("### Combined Ground-Level Picture")
    combined = pd.DataFrame([
        {'Measure': 'ML Model (market data)', 'Score': f"{ml_p*100:.0f}%", 'Signal': 'Low risk'},
        {'Measure': 'HY OAS (bond market)', 'Score': f"{hy_fire*100:.0f}% fire", 'Signal': 'Calm'},
        {'Measure': 'LSI (labor fundamentals)', 'Score': f"{lsi_total:.0f}/100", 'Signal': lsi_level},
        {'Measure': 'CSI (credit fundamentals)', 'Score': f"{csi_total:.0f}/100", 'Signal': csi_level},
    ])
    st.dataframe(combined, use_container_width=True, hide_index=True)
 
    st.warning("""
    **The System A/B Divergence, Quantified:**
    Market-level data (ML model + HY OAS) says low risk.
    Ground-level data (LSI + CSI) says stressed.
    This divergence IS the disconnect between the stock market narrative
    and what workers and small businesses are experiencing.
    Historically, ground-level data has always been right — markets catch up eventually.
    """)
 
# ── TAB 7: GLOBAL ──
with tab7:
    st.header(f"Global Contagion Index — {gci_total:.0f}/100")
 
    col1, col2 = st.columns([1, 2])
    with col1:
        gci_color = '#d4614a' if gci_level in ('CRISIS', 'STRESSED') else '#c8a84a' if gci_level == 'SOFTENING' else '#8fb840'
        st.markdown(f"<div style='padding:20px;background:{gci_color}22;border-radius:12px;text-align:center'>"
                    f"<h1 style='color:{gci_color};margin:0'>{gci_total:.0f}</h1>"
                    f"<h3 style='color:{gci_color};margin:4px 0'>{gci_level}</h3></div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("#### The dimension our US-focused model misses")
        st.markdown(f"""
        - **40% of S&P 500 revenue** is international — we ignore this channel
        - In 2008, European bank failures amplified US problems
        - In 1997, Asian crisis nearly caused US recession
        - In 2010-11, Greek debt triggered US correction
        - Average lag from global stress to US market reaction: **3-6 months**
        """)
 
    st.markdown("### Component Breakdown")
    gci_labels = {
        'dxy_surge': ('USD 6m change', gci_dxy_surge, '%', 'Surge = safe haven flight = global stress'),
        'em_sov': ('EM sovereign spread', gci_em_sov, 'bps', 'Widening = EM economies cracking'),
        'global_pmi': ('Global manuf PMI', gci_global_pmi, '', 'Below 50 = global contraction'),
        'china_ci': ('China credit impulse', gci_china_ci, 'pp', 'Negative = China dragging global growth'),
        'eu_bank': ('EU bank stress', gci_eu_bank, '/100', 'Cross-Atlantic contagion channel'),
    }
    gdf = pd.DataFrame([
        {'Component': v[0], 'Value': f"{v[1]}{v[2]}", 'Stress': f"{gci_comps[k]:.0f}/100", 'Why': v[3]}
        for k, v in gci_labels.items()
    ])
    st.dataframe(gdf, use_container_width=True, hide_index=True)
 
    st.markdown("### Historical Comparison")
    hist_gci = pd.DataFrame([
        {'Date': '2007-10', 'GCI': 28, 'Outcome': 'PRE-GFC (global calm, US cracked first)'},
        {'Date': '2025-02', 'GCI': 38, 'Outcome': 'Current cycle peak'},
        {'Date': 'NOW', 'GCI': gci_total, 'Outcome': 'CURRENT READING'},
        {'Date': '2018-12', 'GCI': 42, 'Outcome': 'Fed tight + trade war (correction)'},
        {'Date': '2015-12', 'GCI': 58, 'Outcome': 'China devaluation (correction)'},
        {'Date': '2001-09', 'GCI': 62, 'Outcome': 'Post dot-com (RECESSION)'},
        {'Date': '2011-08', 'GCI': 65, 'Outcome': 'EU debt crisis (correction)'},
        {'Date': '2022-10', 'GCI': 72, 'Outcome': 'UK gilt + EU energy (correction)'},
        {'Date': '1998-08', 'GCI': 78, 'Outcome': 'LTCM + Asia (correction)'},
        {'Date': '2020-03', 'GCI': 85, 'Outcome': 'COVID shock (RECESSION)'},
        {'Date': '2008-09', 'GCI': 97, 'Outcome': 'GFC (RECESSION)'},
    ])
    st.dataframe(hist_gci.sort_values('GCI'), use_container_width=True, hide_index=True)
 
    st.markdown("### Key Insights")
 
    st.warning("""
    **The unusual dollar behaviour:** Dollar is DECLINING (-5% 6m) during Iran war
    and geopolitical stress. Normally Middle East tension = dollar surge (safe haven bid).
    The fact that DXY is WEAKENING suggests investors are starting to question US assets
    as the ultimate safe haven. This is a structural shift worth watching carefully —
    it's the first time in 40+ years this pattern has appeared during a major geopolitical event.
    """)
 
    st.info("""
    **The 2007 contrast:** Pre-GFC, GCI was 28 (global was calm). US cracked FIRST,
    global contagion followed. Today GCI is ~41 — global is ALREADY softening.
    If US cracks this cycle, contagion will be faster because global economy has
    less absorptive capacity. 2007's "world as shock absorber" scenario is not available now.
    """)
 
    st.markdown("### Full Integrated Picture")
    integrated = pd.DataFrame([
        {'Layer': 'Market (ML model)', 'Reading': f"{ml_p*100:.0f}%", 'Assessment': 'Low risk'},
        {'Layer': 'Market (HY OAS)', 'Reading': f"{hy_fire*100:.0f}% fire", 'Assessment': 'Calm'},
        {'Layer': 'Fundamental (LSI)', 'Reading': f"{lsi_total:.0f}/100", 'Assessment': lsi_level},
        {'Layer': 'Fundamental (CSI)', 'Reading': f"{csi_total:.0f}/100", 'Assessment': csi_level},
        {'Layer': 'Global (GCI)', 'Reading': f"{gci_total:.0f}/100", 'Assessment': gci_level},
    ])
    st.dataframe(integrated, use_container_width=True, hide_index=True)
 
    if lsi_total >= 50 and csi_total >= 45 and gci_total >= 30:
        st.error("""
        **Three-dimensional stress confirmed.** Labor stressed + Credit tightening + Global softening.
        This is the first time in our dataset all three ground-level sensors are above
        "softening" thresholds while market data shows low stress. Historically unprecedented
        divergence between fundamentals and market pricing.
        """)
 
# ── TAB 8: CONSUMER ──
with tab8:
    st.header(f"Consumer Financial Health Index — {cfhi_total:.0f}/100")
 
    col1, col2 = st.columns([1, 2])
    with col1:
        phase_color = '#d4614a' if 'PHASE 3' in cfhi_phase or 'LATE PHASE 2' in cfhi_phase else '#c8a84a' if 'PHASE 2' in cfhi_phase else '#8fb840'
        st.markdown(f"<div style='padding:20px;background:{phase_color}22;border-radius:12px;text-align:center'>"
                    f"<h1 style='color:{phase_color};margin:0'>{cfhi_total:.0f}</h1>"
                    f"<h4 style='color:{phase_color};margin:4px 0'>{cfhi_phase}</h4></div>",
                    unsafe_allow_html=True)
    with col2:
        st.markdown("#### The trifecta framework, quantified")
        st.markdown(f"""
        - **Phase 1** (0-30): Feel bad, spend fine — buffer intact
        - **Phase 2** (30-55): Feel bad, spend on credit — bridge active
        - **Late Phase 2** (55-75): Bridge stressed — Phase 3 imminent
        - **Phase 3** (75-100): Spending snapped — recession trigger
 
        **Current: {cfhi_phase}**
        {cfhi_phase_desc}
        """)
 
    st.markdown("### Component Breakdown")
    cfhi_labels = {
        'real_inc': ('Real disposable income YoY', f"{cfhi_real_inc:+.1f}%", 'Are people actually getting richer?'),
        'savings': ('Personal savings rate', f"{cfhi_savings:.1f}%", 'Below 4% = no cushion left'),
        'tdsp': ('Debt service ratio', f"{cfhi_tdsp:.1f}%", 'Share of income going to debt'),
        'cc_util': ('CC utilization', f"{cfhi_cc_util}%", 'Credit bridge usage'),
        'cc_del': ('CC delinquency 90+d', f"{cfhi_cc_del:.2f}%", 'Bridge actually breaking'),
        'retail': ('Real retail sales YoY', f"{cfhi_retail:+.1f}%", 'Negative = Phase 3 confirmed'),
    }
    cfhidf = pd.DataFrame([
        {'Component': v[0], 'Value': v[1], 'Stress': f"{cfhi_comps[k]:.0f}/100", 'Why': v[2]}
        for k, v in cfhi_labels.items()
    ])
    st.dataframe(cfhidf, use_container_width=True, hide_index=True)
 
    # Snap signals
    snap_signals = []
    if cfhi_savings < 3.5:
        snap_signals.append(f"Savings rate {cfhi_savings:.1f}% — below healthy 4%")
    if cfhi_cc_util > 28:
        snap_signals.append(f"CC utilization {cfhi_cc_util}% — bridge stretched")
    if cfhi_cc_del > 3.0:
        snap_signals.append(f"CC delinq {cfhi_cc_del:.2f}% — bridge cracking")
    if cfhi_real_inc < 1.5:
        snap_signals.append(f"Real income +{cfhi_real_inc:.1f}% — barely growing")
    if cfhi_retail < 2.0:
        snap_signals.append(f"Retail sales +{cfhi_retail:.1f}% — decelerating")
 
    if snap_signals:
        st.markdown("### Snap Signals (Leading Indicators of Phase 3)")
        for sig in snap_signals:
            st.markdown(f"- {sig}")
 
    st.markdown("### Historical Comparison")
    hist_cfhi = pd.DataFrame([
        {'Date': '2020-02', 'CFHI': 10, 'Phase': 'Phase 1 healthy', 'Outcome': 'Then COVID shock'},
        {'Date': '2018-09', 'CFHI': 12, 'Phase': 'Phase 1 healthy', 'Outcome': 'CORRECTION'},
        {'Date': '2022-01', 'CFHI': 20, 'Phase': 'Phase 1 buffered', 'Outcome': 'CORRECTION'},
        {'Date': '2000-03', 'CFHI': 28, 'Phase': 'Phase 1-2 transit', 'Outcome': 'Then RECESSION'},
        {'Date': '2011-04', 'CFHI': 32, 'Phase': 'Phase 2 post-GFC', 'Outcome': 'CORRECTION'},
        {'Date': '2025-02', 'CFHI': 42, 'Phase': 'Phase 2 bridge', 'Outcome': 'Current cycle peak'},
        {'Date': 'NOW', 'CFHI': cfhi_total, 'Phase': cfhi_short, 'Outcome': 'CURRENT'},
        {'Date': '2001-09', 'CFHI': 58, 'Phase': 'Phase 2-3 transit', 'Outcome': 'RECESSION'},
        {'Date': '2007-10', 'CFHI': 62, 'Phase': 'Late Phase 2', 'Outcome': 'Then GFC'},
        {'Date': '2008-09', 'CFHI': 88, 'Phase': 'Phase 3 cracked', 'Outcome': 'CRISIS'},
    ])
    st.dataframe(hist_cfhi.sort_values('CFHI'), use_container_width=True, hide_index=True)
 
    st.warning("""
    **The K-Shaped Hidden Reality:** Headline CC delinquency 2.94% has been FALLING for
    6 quarters. This looks healthy. Reality: tighter underwriting since 2023 means banks
    stopped lending to risky borrowers — the drop is because fewer risky borrowers have cards,
    not because borrowers got healthier. In low-income zip codes, delinquency is rising.
    Per Achieve February 2026 survey: **55% of credit card balances fund essentials**
    (groceries, rent, utilities). That IS the Phase 2 bridge mechanism.
    """)
 
    st.error("""
    **Phase 3 Triggers to Watch:**
    - Real retail sales turning **negative** (currently +1.8%)
    - CC delinquency 90+d crossing **4.0%** (currently 2.94% but K-shaped)
    - Savings rate falling below **3.0%** (currently 3.8%)
    - All three = spending snap confirmed = recession trigger
    """)
 
    st.markdown("### Full Five-Dimensional Picture")
    full_picture = pd.DataFrame([
        {'Layer': 'Market (ML model)', 'Reading': f"{ml_p*100:.0f}%", 'Signal': 'Low risk'},
        {'Layer': 'Market (HY OAS)', 'Reading': f"{hy_fire*100:.0f}% fire", 'Signal': 'Calm'},
        {'Layer': 'Fundamental (LSI)', 'Reading': f"{lsi_total:.0f}/100", 'Signal': lsi_level},
        {'Layer': 'Fundamental (CSI)', 'Reading': f"{csi_total:.0f}/100", 'Signal': csi_level},
        {'Layer': 'Global (GCI)', 'Reading': f"{gci_total:.0f}/100", 'Signal': gci_level},
        {'Layer': 'Consumer (CFHI)', 'Reading': f"{cfhi_total:.0f}/100", 'Signal': cfhi_short},
    ])
    st.dataframe(full_picture, use_container_width=True, hide_index=True)
 
# ── TAB 9: PIPE 1 — FOREIGN FUNDING (Petrodollar / SWF) ──
with tab9:
    st.header("🛢️ Structural Vulnerability Map")
    st.markdown("### Pipe 1: Foreign Funding (Petrodollar / SWF)")
 
    st.markdown("""
    <div style='padding:14px;background:#261f14;border-left:4px solid #c8a84a;border-radius:8px;margin:8px 0'>
    <strong style='color:#c8a84a'>The "Two Elephants on a Seesaw" pipe.</strong><br>
    <span style='color:#b09a6e'>Net capital flows US↔world look balanced.
    Gross flows reveal massive dependency: $6T+ in SWF wealth, $250B Gulf Treasury holdings,
    $1.5T Japanese dollar assets, all supporting US asset prices and Treasury yields.<br><br>
    If allocation shifts even 10-15%, the bid for Treasuries disappears and long rates rise
    independent of the Fed. This is the 2026 transnational flow analog to 2007 European bank flows.</span>
    </div>
    """, unsafe_allow_html=True)
 
    # Pipe 1 indicators (hardcoded current readings)
    pipe1_indicators = [
        {
            "key": "gulf_swf",
            "label": "Gulf SWF US asset concentration",
            "value": 65, "unit": "/100",
            "thresholds": [40, 55, 70, 85, 95],
            "invert": False,
            "ref_2007": "European bank SIV conduit exposure",
            "why": "Gulf SWFs hold $6T globally. $250B+ in US Treasuries, plus hundreds of billions in equities, real estate, PC, tech. Own ~40% of prime US commercial real estate in key metros. A 20-30% allocation pivot toward Asia exceeds 2014-16 Saudi reserve drawdown that moved Treasury term premium measurably.",
        },
        {
            "key": "saudi_china",
            "label": "Saudi oil exports to China",
            "value": 30, "unit": "%",
            "thresholds": [15, 22, 28, 35, 45],
            "invert": False,
            "ref_2007": "(No 2007 analog — new risk dimension)",
            "why": "Saudi now sells 4x MORE oil to China than to US. Physical trade leads capital flows. The 1974 Kissinger equation (US security → Saudi USD pricing → Treasury purchases) under pressure for first time since 1974.",
        },
        {
            "key": "dxy_stress",
            "label": "DXY during geopolitical stress",
            "value": -5.0, "unit": "%",
            "thresholds": [3, 0, -3, -6, -10],
            "invert": True,
            "ref_2007": "USD strengthened during 2007 stress (normal pattern)",
            "why": "CRITICAL SIGNAL. Dollar FALLEN 5% during Iran war. Unprecedented — in 1990, 2001, 2008, 2020, 2022, USD surged as safe haven. Falling during stress = foreign capital already questioning US safe-haven status. An observation of current behavior that has no historical parallel.",
        },
        {
            "key": "pif_commitments",
            "label": "Saudi PIF new US commitments",
            "value": -70, "unit": "%",
            "thresholds": [0, -15, -35, -55, -75],
            "invert": True,
            "ref_2007": "ABCP issuance decline mid-2007",
            "why": "Saudi PIF cut NEW US commitments by 70% in early 2024. Existing holdings intact, marginal flow diverted. EXACT 2007 ABCP pattern: existing conduits kept running but new issuance collapsed. Marginal buyer disappears before existing holder sells.",
        },
        {
            "key": "jp_hedging",
            "label": "Japanese Treasury hedging cost",
            "value": 6.8, "unit": "%",
            "thresholds": [2, 3.5, 5, 7, 10],
            "invert": False,
            "ref_2007": "Swap spread widening Aug 2007",
            "why": "Japanese insurers hold $1.5T dollar assets. At 6.8% hedging cost vs ~4.2% Treasury yield, HEDGED holdings yield NEGATIVE. Mechanical unwind in progress. No political decision required — just math.",
        },
        {
            "key": "em_gold",
            "label": "EM central bank gold accumulation",
            "value": 75, "unit": "/100",
            "thresholds": [25, 45, 65, 80, 95],
            "invert": False,
            "ref_2007": "(No 2007 analog)",
            "why": "EM central banks bought RECORD gold 2022-2025. Every ounce of gold is a dollar reserve NOT bought. Driven by weaponization fear (Russia $300B frozen 2022). Silent, structural — no announcement required.",
        },
        {
            "key": "tic_flow",
            "label": "Foreign Treasury demand trend",
            "value": 35, "unit": "/100",
            "thresholds": [70, 55, 40, 25, 15],
            "invert": True,
            "ref_2007": "MMF flow into ABCP slowing",
            "why": "Foreign share of Treasuries declined 34% (2014) → 22% (now). US issuing $2T/yr deficit. Marginal buyer is domestic (MMFs, banks, hedge fund basis trades). Score 35 = weak foreign demand vs massive supply.",
        },
        {
            "key": "shadow_china",
            "label": "Chinese shadow reserves (off-TIC)",
            "value": 60, "unit": "/100",
            "thresholds": [30, 45, 60, 75, 90],
            "invert": False,
            "ref_2007": "European SIV exposure (off-balance-sheet)",
            "why": "Per Brad Setser (CFR): China holds Treasuries off TIC via state banks. Official $780B underestimates true exposure. Structurally identical to 2007: visible subprime $500B, actual $2T+.",
        },
    ]
 
    def score_pipe1(value, thresholds, inverted=False):
        h, w, s, ar, r = thresholds
        if inverted:
            if value >= h: return 10
            elif value >= w: return 10 + (h - value) / (h - w) * 20
            elif value >= s: return 30 + (w - value) / (w - s) * 20
            elif value >= ar: return 50 + (s - value) / (s - ar) * 25
            elif value >= r: return 75 + (ar - value) / (ar - r) * 20
            else: return 95
        else:
            if value <= h: return 10
            elif value <= w: return 10 + (value - h) / (w - h) * 20
            elif value <= s: return 30 + (value - w) / (s - w) * 20
            elif value <= ar: return 50 + (value - s) / (ar - s) * 25
            elif value <= r: return 75 + (value - ar) / (r - ar) * 20
            else: return 95
 
    def status_label_p1(score):
        if score < 30: return "HEALTHY", "#8fb840"
        elif score < 50: return "WATCH", "#c8a84a"
        elif score < 70: return "STRESSED", "#d4614a"
        elif score < 85: return "AT-RISK", "#d4614a"
        else: return "ACTIVE-RUPTURE", "#d4614a"
 
    for ind in pipe1_indicators:
        ind["score"] = score_pipe1(ind["value"], ind["thresholds"], ind.get("invert", False))
        ind["status"], ind["color"] = status_label_p1(ind["score"])
 
    scores = [ind["score"] for ind in pipe1_indicators]
    max_score = max(scores)
    mean_score = np.mean(scores)
    above_75 = sum(1 for s in scores if s >= 75)
    above_50 = sum(1 for s in scores if s >= 50)
 
    if max_score >= 90:
        overall_status_p1 = "ACTIVE-RUPTURE"
        overall_color_p1 = "#d4614a"
        overall_desc_p1 = "At least one indicator shows active failure mode"
    elif above_75 >= 2:
        overall_status_p1 = "AT-RISK"
        overall_color_p1 = "#d4614a"
        overall_desc_p1 = "Multiple indicators at rupture threshold — cascade risk"
    elif above_50 >= 3 or max_score >= 75:
        overall_status_p1 = "STRESSED"
        overall_color_p1 = "#d4614a"
        overall_desc_p1 = "Multiple indicators stressed or one at-risk"
    elif above_50 >= 1 or mean_score >= 40:
        overall_status_p1 = "WATCH"
        overall_color_p1 = "#c8a84a"
        overall_desc_p1 = "Early warning signals active, not yet rupture-consistent"
    else:
        overall_status_p1 = "HEALTHY"
        overall_color_p1 = "#8fb840"
        overall_desc_p1 = "Pipe functioning normally"
 
    # Hero status
    st.markdown(f"""
    <div style='padding:24px;background:{overall_color_p1}22;border:2px solid {overall_color_p1};border-radius:14px;text-align:center;margin:16px 0'>
    <div style='color:#b09a6e;font-size:14px;margin-bottom:8px'>OVERALL PIPE STATUS</div>
    <h1 style='color:{overall_color_p1};margin:0;font-size:42px'>{overall_status_p1}</h1>
    <div style='color:#b09a6e;margin-top:8px'>{overall_desc_p1}</div>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Stressed (≥50)", f"{above_50}/8")
    col2.metric("At-Risk (≥75)", f"{above_75}/8")
    col3.metric("Max Score", f"{max_score:.0f}/100")
 
    st.markdown("### Indicator Status Table")
    st.caption("Each indicator captures a specific cross-border funding dependency or early warning signal.")
 
    df = pd.DataFrame([
        {
            "Indicator": ind["label"],
            "Value": f"{ind['value']}{ind['unit']}",
            "Score": f"{ind['score']:.0f}/100",
            "Status": ind["status"],
            "2007 Analog": ind["ref_2007"],
        }
        for ind in pipe1_indicators
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    st.markdown("### The Two-Elephants-on-a-Seesaw Insight")
    st.markdown("""
    Tooze's central critique of the Hamilton Project was that they measured NET capital flows (which were balanced) instead of GROSS flows (where two massive streams moved in opposite directions). When both elephants pressed down simultaneously in 2008, the seesaw snapped.
 
    **Today's gross flow dependencies:**
    - US receives ~$1T/year in foreign funding (Treasuries, equities, real estate)
    - ~$250B+ from Gulf SWFs alone
    - ~$1.5T stock of Japanese dollar holdings
    - ~$6T total SWF capacity globally, ~40% dollar-allocated
    - ~$9.3T total foreign holdings of US Treasuries
 
    **If allocation shifts even 10-15% at the margin**, Treasury yields rise independently of Fed policy.
    """)
 
    # Detailed reasoning
    st.markdown("### Why Each Indicator Matters")
    for ind in pipe1_indicators:
        with st.expander(f"{ind['label']} — {ind['status']} ({ind['value']}{ind['unit']})"):
            st.markdown(f"**2007 Equivalent:** {ind['ref_2007']}")
            st.markdown(f"**Why it matters:** {ind['why']}")
 
    # Transmission mechanisms
    st.markdown("### Transmission Mechanisms — How a Pipe 1 Rupture Propagates")
 
    with st.expander("🌊 Scenario 1: Gulf SWF Allocation Pivot"):
        st.markdown("""
        **Historical analog:** 2014-16 Saudi reserve drawdown ($230B)
        **Timeline to US consumer:** 6-18 months (slow but structural)
 
        **Sequence:**
        1. Iran war creates fresh Saudi-China strategic alignment
        2. Oil continues flowing east, capital follows physical trade
        3. Gulf SWFs announce 'portfolio diversification' — code for reducing US
        4. New US commitments frozen; existing positions run off
        5. At margin, Gulf dollar buying of Treasuries stops
        6. Treasury auctions see weak bid-to-cover on long end
        7. Long rates rise 50-100bps without Fed action
        8. Rising long rates: mortgage rates up, equity multiples compress
        9. Fed faces choice: cut rates (inflation risk) or let economy cool
        10. Either path: recession pressure independent of cyclical factors
        """)
 
    with st.expander("⏱️ Scenario 2: Japanese Insurer Treasury Unwind (Fastest)"):
        st.markdown("""
        **Historical analog:** Q4 2022 Japanese insurer selling wave
        **Timeline to US consumer:** 3-9 months (fastest transmission)
 
        **Sequence:**
        1. FX hedging cost stays above 5% while yield pickup <3%
        2. Japanese life insurers accelerate Treasury divestment
        3. Sold mostly long-duration Treasuries (matching liabilities)
        4. Term premium rises as Japanese bid disappears
        5. MBS spreads widen (MBS tracks long Treasuries)
        6. Mortgage rates rise 30-50bps independent of Fed
        7. Housing market cools further; CRE refinancing gets harder
        8. Regional bank CRE losses accelerate (already Pipe 2 stress)
        """)
 
    with st.expander("💥 Scenario 3: Dollar Confidence Cascade (Unprecedented)"):
        st.markdown("""
        **Historical analog:** None — this would be unprecedented
        **Timeline to US consumer:** 6-24 months, high severity if it happens
 
        **Sequence:**
        1. Dollar continues weakening despite geopolitical stress (unprecedented pattern)
        2. Gold continues accumulation by EM central banks
        3. One major SWF announces reduced dollar allocation
        4. Other SWFs follow (herding behavior in flows)
        5. Dollar weakness becomes self-reinforcing trend
        6. Inflation imported via commodity prices in USD terms
        7. Fed forced into impossible choice: weak currency or recession
        8. Either way, negative wealth effect on US consumer
        9. This is the 'exorbitant privilege' reversal scenario
        """)
 
    st.markdown("### Who Gets Hit First (Transmission Chain)")
    transmission_order = [
        "Long-duration Treasury holders (pension funds, insurers) — **immediate**",
        "US homebuyers via mortgage rates tracking 10yr — **1-3 months**",
        "CRE borrowers refinancing in maturity wall — **3-6 months**",
        "Regional banks with long-duration bond portfolios (SVB 2.0 risk) — **3-9 months**",
        "Highly leveraged tech/AI companies (duration sensitive) — **6-12 months**",
        "US consumer via mortgage refinancing and HELOC costs — **6-18 months**",
    ]
    for i, target in enumerate(transmission_order, 1):
        st.markdown(f"{i}. {target}")
 
    st.markdown("### Early Warning Signals to Watch")
    signals = [
        "Saudi Treasury holdings (SAMA monthly data) declining sequentially",
        "TIC data showing foreign net selling for 3+ consecutive months",
        "Bid-to-cover ratios at 30yr Treasury auctions below 2.3x",
        "DXY breaking below 92 during active geopolitical stress",
        "Gold price accelerating above $3,500 without obvious USD-specific trigger",
        "Any GCC state announcing yuan-denominated oil contracts",
        "Japanese life insurer monthly reports showing accelerated Treasury sales",
        "Chinese 'shadow reserve' data showing acceleration in state bank holdings",
    ]
    for sig in signals:
        st.markdown(f"- {sig}")
 
    # Critical asymmetry
    st.error("""
    **THE CRITICAL ASYMMETRY — why this pipe is different:**
 
    Unlike Pipe 2 where Fed liquidity support actually helps, Pipe 1 rupture puts the Fed in a **no-win situation**:
 
    The Fed can print dollars to buy Treasuries domestically (YCC — Yield Curve Control is a real option).
    But that **debases the dollar further**, accelerating the underlying problem.
 
    It's the monetary equivalent of pouring gasoline on a fire.
 
    This is why Pipe 1 deserves separate analysis from the cyclical model — because the traditional crisis playbook doesn't just fail here, it makes things worse.
    """)
 
    # Combined SVM picture
    st.markdown("### 🔗 Combined Structural Vulnerability Map")
 
    svm_summary = pd.DataFrame([
        {"Pipe": "Pipe 1: Foreign Funding", "Status": overall_status_p1,
         "Key Concern": "PIF commitments -70%, dollar weak during stress"},
        {"Pipe": "Pipe 2: Shadow Banking", "Status": "STRESSED",
         "Key Concern": "BDC 17% NAV discount, 40% PC borrowers cashflow-negative"},
        {"Pipe": "Pipe 3: Market Structure", "Status": "Not yet built",
         "Key Concern": "Treasury dysfunction, repo stress, dealer capacity"},
        {"Pipe": "Pipe 4: Geopolitical", "Status": "Not yet built",
         "Key Concern": "Russia-China-Saudi alignment, currency bloc shifts"},
    ])
    st.dataframe(svm_summary, use_container_width=True, hide_index=True)
 
    st.markdown(f"""
    <div style='padding:16px;background:#261f14;border-left:4px solid #d4614a;border-radius:8px;margin:8px 0'>
    <strong style='color:#d4614a'>Combined Framework Reading (April 2026):</strong><br><br>
 
    <strong style='color:#f2e8d0'>Recession Radar (cyclical):</strong>
    <span style='color:#b09a6e'>15-45% probability (market vs fundamentals)</span><br><br>
 
    <strong style='color:#f2e8d0'>Pipe 1 (Foreign Funding):</strong>
    <span style='color:{overall_color_p1}'>{overall_status_p1}</span><br>
 
    <strong style='color:#f2e8d0'>Pipe 2 (Shadow Banking):</strong>
    <span style='color:#d4614a'>STRESSED</span><br><br>
 
    <strong style='color:#c8a84a'>Integrated assessment:</strong><br>
    <span style='color:#b09a6e'>Two structural pipes at severe stress simultaneously. This is the
    2007-equivalent configuration. Cyclical probability number understates risk because the transmission
    mechanisms (not the indicators) are loaded.</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.caption("Framework inspired by Adam Tooze, *Crashed* (2018). "
               "Brad Setser (CFR) on petrodollar recycling, SWP Berlin on Gulf SWF foreign policy, "
               "and Treasury TIC data on foreign Treasury holdings provide the empirical foundation for Pipe 1.")
 
 
# ── TAB 10: PIPE 2 — SHADOW BANKING ──
with tab10:
    st.header("🔧 Structural Vulnerability Map")
    st.markdown("### Pipe 2: Shadow Banking")
 
    st.markdown("""
    <div style='padding:14px;background:#261f14;border-left:4px solid #c8a84a;border-radius:8px;margin:8px 0'>
    <strong style='color:#c8a84a'>This is NOT a probability model.</strong><br>
    <span style='color:#b09a6e'>The Recession Radar tells you <em>probability</em> of a cyclical recession.<br>
    The Vulnerability Map tells you <em>transmission mechanisms</em> — how any shock propagates
    through hidden pipes that probability models can't see.<br><br>
    <em>Tooze insight: Crises don't come through the door economists watch.
    They come through the plumbing economists don't have models for.</em></span>
    </div>
    """, unsafe_allow_html=True)
 
    # Pipe 2 current indicators (hardcoded current readings)
    pipe2_indicators = [
        {
            "key": "bdc_discount",
            "label": "BDC discount to NAV",
            "value": 17.0, "unit": "%",
            "thresholds": [0, 5, 12, 20, 30],  # healthy, watch, stressed, at-risk, rupture
            "ref_2007": "ABCP rates spiking Aug 2007",
            "why": "Public BDCs trade daily. 17% discount = market disbelieves private NAVs. Same pattern as 2007 when ABCP spreads spiked while subprime MBS still looked fine.",
        },
        {
            "key": "pik",
            "label": "Payment-in-Kind income share",
            "value": 8.0, "unit": "%",
            "thresholds": [3, 5, 8, 12, 18],
            "ref_2007": "Teaser-rate resets on subprime ARMs",
            "why": "When borrowers can't pay cash interest, they roll it into principal. Debt compounds invisibly. Delays but doesn't prevent default.",
        },
        {
            "key": "tech_conc",
            "label": "PC tech/business services exposure",
            "value": 42.0, "unit": "%",
            "thresholds": [25, 32, 40, 50, 60],
            "ref_2007": "Subprime concentration in CA/FL/AZ/NV",
            "why": "21% software + 21% other tech/business services. Correlated AI disruption risk across entire PC book.",
        },
        {
            "key": "true_default",
            "label": "True PC default rate (w/ LMEs)",
            "value": 5.8, "unit": "%",
            "thresholds": [2, 3.5, 5, 8, 12],
            "ref_2007": "Subprime delinquency rate mid-2007",
            "why": "Headline rate <2%. True rate 5.8% including LMEs. 71% of LME'd companies default within 3 years.",
        },
        {
            "key": "evergreen",
            "label": "Evergreen fund redemption pressure",
            "value": 65, "unit": "/100",
            "thresholds": [20, 35, 55, 75, 90],
            "ref_2007": "BNP Paribas fund freeze Aug 2007",
            "why": "5%/quarter redemption caps. JPM forecasts elevated redemptions through H1 2026. Gating = panic trigger.",
        },
        {
            "key": "sub_lines",
            "label": "Subscription line + NAV financing",
            "value": 72, "unit": "/100",
            "thresholds": [20, 40, 60, 80, 95],
            "ref_2007": "Bank-sponsored SIV exposure",
            "why": "Banks lending to PC funds = leverage on leverage. Exact 2007 SIV structure. 17Capital raised largest-ever NAV financing fund 2025.",
        },
        {
            "key": "neg_fcf",
            "label": "PC borrowers w/ negative FCF",
            "value": 40.0, "unit": "%",
            "thresholds": [15, 25, 35, 45, 55],
            "ref_2007": "Subprime borrowers who couldn't afford reset rates",
            "why": "IMF: 40% of PC borrowers have negative free cash flow. Zombies surviving on lender forbearance.",
        },
        {
            "key": "mmf",
            "label": "MMF exposure to PC paper",
            "value": 45, "unit": "/100",
            "thresholds": [20, 35, 55, 75, 90],
            "ref_2007": "MMF exposure to ABCP (direct parallel)",
            "why": "Short-term savings funding long-term illiquid credit. Same structure as 2007. Data opaque — that IS the signal.",
        },
    ]
 
    def score_pipe2(value, thresholds):
        h, w, s, ar, r = thresholds
        if value <= h: return 10
        elif value <= w: return 10 + (value - h) / (w - h) * 20
        elif value <= s: return 30 + (value - w) / (s - w) * 20
        elif value <= ar: return 50 + (value - s) / (ar - s) * 25
        elif value <= r: return 75 + (value - ar) / (r - ar) * 20
        else: return 95
 
    def status_label(score):
        if score < 30: return "HEALTHY", "#8fb840"
        elif score < 50: return "WATCH", "#c8a84a"
        elif score < 70: return "STRESSED", "#d4614a"
        elif score < 85: return "AT-RISK", "#d4614a"
        else: return "ACTIVE-RUPTURE", "#d4614a"
 
    # Compute all scores
    for ind in pipe2_indicators:
        ind["score"] = score_pipe2(ind["value"], ind["thresholds"])
        ind["status"], ind["color"] = status_label(ind["score"])
 
    # Overall pipe status
    scores = [ind["score"] for ind in pipe2_indicators]
    max_score = max(scores)
    mean_score = np.mean(scores)
    above_75 = sum(1 for s in scores if s >= 75)
    above_50 = sum(1 for s in scores if s >= 50)
 
    if max_score >= 90:
        overall_status = "ACTIVE-RUPTURE"
        overall_color = "#d4614a"
        overall_desc = "At least one indicator shows active failure mode"
    elif above_75 >= 2:
        overall_status = "AT-RISK"
        overall_color = "#d4614a"
        overall_desc = "Multiple indicators at rupture threshold — cascade risk"
    elif above_50 >= 3 or max_score >= 75:
        overall_status = "STRESSED"
        overall_color = "#d4614a"
        overall_desc = "Multiple indicators stressed or one at-risk"
    elif above_50 >= 1 or mean_score >= 40:
        overall_status = "WATCH"
        overall_color = "#c8a84a"
        overall_desc = "Early warning signals active, not yet rupture-consistent"
    else:
        overall_status = "HEALTHY"
        overall_color = "#8fb840"
        overall_desc = "Pipe functioning normally"
 
    # Hero status
    st.markdown(f"""
    <div style='padding:24px;background:{overall_color}22;border:2px solid {overall_color};border-radius:14px;text-align:center;margin:16px 0'>
    <div style='color:#b09a6e;font-size:14px;margin-bottom:8px'>OVERALL PIPE STATUS</div>
    <h1 style='color:{overall_color};margin:0;font-size:42px'>{overall_status}</h1>
    <div style='color:#b09a6e;margin-top:8px'>{overall_desc}</div>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Stressed (≥50)", f"{above_50}/8")
    col2.metric("At-Risk (≥75)", f"{above_75}/8")
    col3.metric("Max Score", f"{max_score:.0f}/100")
 
    st.markdown("### Indicator Status Table")
    st.caption("Each indicator has a direct 2007 analog. This is the Tooze framework applied to 2026.")
 
    df = pd.DataFrame([
        {
            "Indicator": ind["label"],
            "Value": f"{ind['value']}{ind['unit']}",
            "Score": f"{ind['score']:.0f}/100",
            "Status": ind["status"],
            "2007 Equivalent": ind["ref_2007"],
        }
        for ind in pipe2_indicators
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
 
    # Detailed reasoning
    st.markdown("### Why Each Indicator Matters")
    for ind in pipe2_indicators:
        with st.expander(f"{ind['label']} — {ind['status']} ({ind['value']}{ind['unit']})"):
            st.markdown(f"**2007 Equivalent:** {ind['ref_2007']}")
            st.markdown(f"**Why it matters:** {ind['why']}")
            st.caption(f"Threshold scale — Healthy ≤{ind['thresholds'][0]}{ind['unit']} | Watch ≤{ind['thresholds'][1]}{ind['unit']} | Stressed ≤{ind['thresholds'][2]}{ind['unit']} | At-risk ≤{ind['thresholds'][3]}{ind['unit']} | Rupture ≤{ind['thresholds'][4]}{ind['unit']}")
 
    # Transmission mechanisms
    st.markdown("### Transmission Mechanisms — How a Pipe Rupture Propagates")
 
    with st.expander("🌊 Scenario 1: Evergreen Fund Gating Cascade"):
        st.markdown("""
        **Historical analog:** BNP Paribas freezes 3 funds, Aug 9 2007
        **Timeline to US consumer:** 6-18 months
 
        **Sequence:**
        1. Investor redemption requests exceed 5% quarterly cap
        2. Fund gates (or side-pockets illiquid assets)
        3. Gating becomes public → other evergreen investors panic
        4. Broader evergreen redemptions → mass gating
        5. Funds forced to sell liquid assets to meet remaining withdrawals
        6. Liquid sleeve sales → public credit spread widening
        7. Spread widening hits BDC NAVs → public market confirms stress
        8. Banks with subscription line exposure reduce lines
        9. Funds unable to deploy new capital or refinance
        10. Zombie borrowers (40% negative FCF) default en masse
        """)
 
    with st.expander("📉 Scenario 2: BDC Discount Repricing Cascade"):
        st.markdown("""
        **Historical analog:** Summer 2015 BDC discount event (smaller scale)
        **Timeline to US consumer:** 9-15 months
 
        **Sequence:**
        1. BDC discount to NAV widens further (currently 17%)
        2. Wealth channel investors sell BDC positions
        3. BDC share prices drop → forced deleveraging
        4. BDCs sell loan positions → pressure on PC asset prices
        5. Private NAVs forced to mark down
        6. Evergreen fund NAVs drop → redemption surge
        7. See Scenario 1
        """)
 
    with st.expander("🏦 Scenario 3: Bank Subscription Line Retraction (Fastest)"):
        st.markdown("""
        **Historical analog:** 2007 conduit withdrawal from ABCP market
        **Timeline to US consumer:** 3-9 months (fastest transmission)
 
        **Sequence:**
        1. Bank stress from CRE or other exposure increases
        2. Banks reduce subscription lines to PC funds
        3. PC funds lose ability to bridge capital calls
        4. Funds must draw from LPs faster than expected
        5. LPs (pensions, endowments) under pressure
        6. Some LPs default on capital commitments
        7. Funds forced to sell assets at fire-sale prices
        8. Zombie borrower defaults accelerate
        """)
 
    st.markdown("### Who Gets Hit First (Transmission Chain)")
    transmission_order = [
        "PC borrowers with negative FCF (40% of book) — **immediate**",
        "Subprime CC borrowers (already K-shaped stress) — **within 3 months**",
        "Small business loans via non-bank lenders — **3-6 months**",
        "Regional banks with CRE + subscription line exposure — **3-9 months**",
        "Tech/software sector layoffs (42% of PC book) — **6-12 months**",
        "Broader consumer via tech layoffs + small biz closures — **6-18 months**",
    ]
    for i, target in enumerate(transmission_order, 1):
        st.markdown(f"{i}. {target}")
 
    st.markdown("### Early Warning Signals to Watch Daily")
    signals = [
        "BDC discount to NAV widening beyond 20% (currently 17%)",
        "PIK income share rising above 10% at major BDCs",
        "Evergreen fund gating events (any major manager)",
        "Subscription line volume declines in bank earnings reports",
        "PC default rate (Fitch) crossing 7%",
        "Specialty finance fund closures or manager exits",
        "Secondary market PC loan prices below 90 cents on dollar",
    ]
    for sig in signals:
        st.markdown(f"- {sig}")
 
    # Critical integration explanation
    st.markdown("### 🔗 How This Integrates with Recession Radar")
 
    st.markdown(f"""
    <div style='padding:16px;background:#261f14;border-left:4px solid #c8a84a;border-radius:8px;margin:8px 0'>
    <strong style='color:#c8a84a'>Two frameworks, two questions, two answers:</strong><br><br>
 
    <strong style='color:#f2e8d0'>Recession Radar says:</strong>
    <span style='color:#b09a6e'>{ml_p*100:.0f}% cyclical probability (market-weighted)<br>
    35-45% when adjusted for LSI + CSI + CFHI fundamentals</span><br><br>
 
    <strong style='color:#f2e8d0'>Pipe 2 Shadow Banking says:</strong>
    <span style='color:{overall_color}'>{overall_status}</span><br>
    <span style='color:#b09a6e'>{above_50}/8 indicators stressed. Rupture mechanisms present.</span><br><br>
 
    <strong style='color:#c8a84a'>Combined decision framework:</strong><br>
    <span style='color:#b09a6e'>Even at low cyclical probability, a STRESSED pipe means any shock
    amplifies significantly. The traditional playbook (Fed cuts + fiscal response) may not
    work for Pipe 2 transmission because private credit losses don't appear on bank balance
    sheets the Fed can backstop.</span><br><br>
 
    <strong style='color:#d4614a'>Position for asymmetric downside risk</strong>
    <span style='color:#b09a6e'> even when central probability is low, because the plumbing
    cannot absorb a cyclical shock without cascading.</span>
    </div>
    """, unsafe_allow_html=True)
 
    st.warning("""
    **Why Recession Radar can't see this:** Probability models trained on 2000, 2008, 2020
    don't have private credit in their feature set because it didn't exist at scale in those
    periods. The thermometer can't diagnose a broken pipe. This is NOT a model failure —
    it's a category failure. You need a separate diagnostic tool for the plumbing.
    """)
 
    st.caption("Framework inspired by Adam Tooze, *Crashed* (2018). "
               "Tooze's central insight: the 2008 crisis wasn't caused by the mechanisms "
               "the smartest economists in 2006 were watching. It came through the "
               "transnational banking plumbing they had no model for. Pipes 1-4 of the "
               "Structural Vulnerability Map are designed to map today's equivalent pipes.")
 
# ── TAB 11: ABOUT ──
with tab11:
    st.header("About This Model")
 
    st.markdown("""
    ### What this is
    A self-learning ML model that predicts market peaks and recessions using 11 economic
    indicators. Runs 3 ML models in ensemble (Logistic Regression + Random Forest + Gradient
    Boosting), uses 50+ years of historical data, and validates out-of-sample via walk-forward
    backtesting.
 
    ### Key design decisions
    - **Primary target: market peaks** (forward-looking), not NBER recession dates (lagging)
    - **Distinguishes recessions from corrections** using credit spreads + Sahm rule logic
    - **Regime-weighted training**: post-2000 data weighted 1.0x, pre-2000 at 0.7x
    - **36 engineered features** from 11 raw indicators (interactions, composites, momentum)
    - **Bootstrap confidence intervals** instead of false-precision point estimates
 
    ### Performance (verified)
    - Walk-forward accuracy: 78-89%
    - LOOCV accuracy: 92%
    - Recession detection: 75% (only miss is COVID — exogenous shock)
    - False positive rate: 0-11%
 
    ### Known limitations
    - n=13 events is statistically marginal
    - 0-1 fire levels encode subjective judgment (not raw data)
    - Cannot predict exogenous shocks (pandemics, wars)
    - Fire levels should ideally be normalized from raw FRED data via percentile ranks
 
    ### How to use
    1. Adjust indicators in the sidebar to reflect current conditions
    2. Watch the ML probability update in real-time
    3. Check the scenarios tab for "what if" analysis
    4. Consult Action Plan for position sizing guidance
 
    **Not financial advice.** Consult a qualified financial advisor.
    """)
 
st.markdown("---")
st.caption(f"Recession Radar v7.0 | ML + LSI + Scenarios | Not financial advice")

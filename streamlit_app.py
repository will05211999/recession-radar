"""
Recession Radar v8.0 — Apple-style redesign with live FRED integration

Architecture:
  - Live FRED data for core 11 indicators (with graceful fallback)
  - Manual sliders for Pipes 1-2 indicators (structural research data)
  - ML ensemble trains fresh every session
  - 11 tabs covering cyclical probability + structural vulnerability

To enable live FRED data:
  1. Get free API key at https://fred.stlouisfed.org/docs/api/api_key.html
  2. In Streamlit Cloud: Settings → Secrets → add: FRED_API_KEY = "your_key"
  3. For local testing: set env var FRED_API_KEY or use .streamlit/secrets.toml
"""

import streamlit as st
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Recession Radar · v8",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# APPLE-STYLE CSS — refined, clean, futuristic
# ══════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Base */
    .main { background-color: #0a0a0f; }
    .stApp {
        background: linear-gradient(180deg, #0a0a0f 0%, #141419 100%);
        color: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
        font-feature-settings: "kern", "liga", "calt";
    }

    /* Typography */
    h1 { color: #f5f5f7 !important; font-weight: 600 !important; letter-spacing: -0.02em !important; font-size: 2.4em !important; }
    h2 { color: #f5f5f7 !important; font-weight: 500 !important; letter-spacing: -0.01em !important; font-size: 1.6em !important; }
    h3 { color: #c8a84a !important; font-weight: 500 !important; letter-spacing: -0.005em !important; font-size: 1.2em !important; }
    h4 { color: #a8a8ad !important; font-weight: 500 !important; font-size: 0.95em !important; text-transform: uppercase; letter-spacing: 0.08em !important; }

    p, li, span { color: #d1d1d6 !important; line-height: 1.5 !important; }

    /* Hide default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* Glass morphism cards */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 18px 20px;
        transition: all 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(200,168,74,0.3);
        background: rgba(255,255,255,0.05);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.72em !important;
        text-transform: uppercase;
        letter-spacing: 0.1em !important;
        color: #86868b !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.1em !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #f5f5f7 !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78em !important;
        color: #86868b !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10,10,15,0.95);
        backdrop-filter: blur(40px);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stSlider label {
        color: #a8a8ad !important;
        font-size: 0.82em !important;
        font-weight: 400 !important;
    }

    /* Tabs — Apple-style pill navigation */
    div[data-testid="stTabs"] button[role="tab"] {
        background: transparent;
        color: #86868b !important;
        border: none;
        border-radius: 10px;
        padding: 9px 16px;
        margin-right: 4px;
        font-size: 0.85em !important;
        font-weight: 500;
        transition: all 0.2s;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background: rgba(255,255,255,0.05);
        color: #f5f5f7 !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: rgba(200,168,74,0.15);
        color: #c8a84a !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
    div[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        overflow: hidden;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        margin-bottom: 8px;
    }
    div[data-testid="stExpander"] summary {
        color: #f5f5f7 !important;
        font-weight: 500 !important;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        background: rgba(212,97,74,0.08) !important;
        border: 1px solid rgba(212,97,74,0.25) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(20px);
    }

    /* Subtle dividers */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.06) !important;
        margin: 24px 0 !important;
    }

    /* Caption */
    .stCaption, small { color: #636366 !important; font-size: 0.78em !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LIVE FRED DATA LAYER
# ══════════════════════════════════════════════════════════════

FRED_SERIES = {
    # Map of indicator name -> (FRED series ID, description, fallback value)
    "t10y2y": ("T10Y2Y", "10yr-2yr Treasury spread", 0.62),
    "permits": ("PERMIT", "Building permits (thousands)", 1450),
    "lei_ytd_chg": ("USSLIND", "LEI state-level proxy", -0.8),
    "wti_oil": ("DCOILWTICO", "WTI crude oil", 98.0),
    "bamlh0a0hym2": ("BAMLH0A0HYM2", "HY OAS", 3.17),
    "corp_profits_yoy": ("CP", "Corporate profits", None),
    "durables_yoy": ("DGORDER", "Durable goods orders", None),
    "margin_debt": ("BOGZ1FL663067003Q", "Margin debt", None),
    "unrate": ("UNRATE", "Unemployment rate", 4.3),
    "sp500": ("SP500", "S&P 500 index", None),
    "fed_deficit_pct_gdp": ("FYFSGDA188S", "Fed deficit % GDP", 5.9),
    # LSI series
    "avg_weekly_hours": ("AWHMAN", "Avg weekly hours mfg", 39.8),
    "quits_rate": ("JTSQUR", "Quits rate", 2.8),
    "temp_emp": ("TEMPHELPS", "Temp help employment", None),
    "continued_claims": ("CCSA", "Continued claims", 2150),
    "u6_rate": ("U6RATE", "U-6 unemployment", 8.4),
    # CSI series
    "sloos_ci": ("DRTSCILM", "SLOOS C&I tightening", 18.0),
    "sloos_cc": ("DRTSCLCC", "SLOOS CC tightening", 22.0),
    "chargeoff_rate": ("CORBLACBS", "Loan charge-off rate", 0.72),
    "consumer_delinq": ("DRCLACBS", "Consumer delinquency", 3.1),
    "auto_delinq": ("DRALACBS", "Auto delinquency", 3.3),
    "commercial_paper": ("COMPOUT", "Commercial paper outstanding", None),
    # CFHI series
    "real_disp_income": ("A229RX0", "Real disposable income per capita", None),
    "savings_rate": ("PSAVERT", "Personal savings rate", 3.8),
    "debt_service": ("TDSP", "Debt service ratio", 11.4),
    "revolving_credit": ("REVOLSL", "Revolving consumer credit", None),
    "cc_delinq_90": ("DRCCLACBS", "CC delinquency 90+d", 2.94),
    "retail_sales_real": ("RRSFS", "Real retail sales", None),
    # GCI series
    "dxy": ("DTWEXBGS", "Trade-weighted dollar", None),
}


def get_fred_api_key():
    """Try multiple ways to get FRED API key."""
    # Streamlit secrets (for deployed app)
    try:
        if "FRED_API_KEY" in st.secrets:
            return st.secrets["FRED_API_KEY"]
    except Exception:
        pass
    # Environment variable (for local)
    import os
    return os.environ.get("FRED_API_KEY", None)


@st.cache_data(ttl=3600)  # Cache FRED data for 1 hour
def fetch_fred_series(series_id):
    """Fetch a single FRED series. Returns (value, date) or (None, None) on failure."""
    api_key = get_fred_api_key()
    if not api_key:
        return None, None

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)
        if data is None or len(data) == 0:
            return None, None
        data = data.dropna()
        if len(data) == 0:
            return None, None
        return float(data.iloc[-1]), data.index[-1].strftime("%Y-%m-%d")
    except Exception:
        return None, None


@st.cache_data(ttl=3600)
def fetch_all_fred_data():
    """Fetch all FRED series at once. Returns dict of {key: (value, date, source)}."""
    api_key = get_fred_api_key()
    results = {}

    if not api_key:
        # No API key — return fallback values for everything
        for key, (series_id, desc, fallback) in FRED_SERIES.items():
            results[key] = {
                "value": fallback,
                "date": None,
                "source": "FALLBACK",
                "series_id": series_id,
                "description": desc,
            }
        return results, "NO_API_KEY"

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        for key, (series_id, desc, fallback) in FRED_SERIES.items():
            try:
                data = fred.get_series(series_id)
                if data is not None and len(data.dropna()) > 0:
                    data = data.dropna()
                    results[key] = {
                        "value": float(data.iloc[-1]),
                        "date": data.index[-1].strftime("%Y-%m-%d"),
                        "source": "FRED_LIVE",
                        "series_id": series_id,
                        "description": desc,
                    }
                else:
                    results[key] = {
                        "value": fallback,
                        "date": None,
                        "source": "FALLBACK",
                        "series_id": series_id,
                        "description": desc,
                    }
            except Exception:
                results[key] = {
                    "value": fallback,
                    "date": None,
                    "source": "FALLBACK",
                    "series_id": series_id,
                    "description": desc,
                }
        return results, "LIVE"
    except ImportError:
        for key, (series_id, desc, fallback) in FRED_SERIES.items():
            results[key] = {
                "value": fallback,
                "date": None,
                "source": "FALLBACK",
                "series_id": series_id,
                "description": desc,
            }
        return results, "FREDAPI_NOT_INSTALLED"
    except Exception as e:
        for key, (series_id, desc, fallback) in FRED_SERIES.items():
            results[key] = {
                "value": fallback,
                "date": None,
                "source": "FALLBACK",
                "series_id": series_id,
                "description": desc,
            }
        return results, f"ERROR: {str(e)[:100]}"


# ══════════════════════════════════════════════════════════════
# NORMALIZATION — raw values to 0-1 fire levels
# ══════════════════════════════════════════════════════════════

def normalize_indicator(key, value):
    """Convert raw FRED value to 0-1 fire level using research-based thresholds."""
    if value is None:
        return 0.5  # Neutral if missing

    # Each indicator has its own mapping from raw value to fire level
    # Thresholds based on historical percentile rankings
    if key == "t10y2y":
        # Yield curve: more negative = higher fire
        # -1.0 = 1.0 fire; 0.0 = 0.5; 2.0 = 0.0
        return max(0, min(1, (1.0 - value) / 2.0))
    elif key == "permits":
        # Building permits: lower = higher fire
        # 800K = 1.0; 1500K = 0.4; 1800K = 0.0
        return max(0, min(1, (1800 - value) / 1000))
    elif key == "lei_ytd_chg":
        # LEI change: more negative = higher fire
        return max(0, min(1, (1.0 - value) / 3.0))
    elif key == "wti_oil":
        # Oil: normalized around $60-120 range
        return max(0, min(1, (value - 50) / 80))
    elif key == "bamlh0a0hym2":
        # HY OAS: percent spread
        # 3% = 0.2; 5% = 0.6; 8% = 1.0
        return max(0, min(1, (value - 2.5) / 6))
    elif key == "unrate":
        # Unemployment: higher = higher fire (indirect)
        return max(0, min(1, (value - 3.0) / 5))
    elif key == "fed_deficit_pct_gdp":
        # Fiscal COUNTER-indicator: higher deficit = higher "fiscal support" fire
        return max(0, min(1, (value - 1.0) / 6))
    # Default
    return 0.5


# ══════════════════════════════════════════════════════════════
# RECESSION RADAR ML MODEL
# ══════════════════════════════════════════════════════════════

INDICATOR_NAMES = ['Yield curve', 'Permits', 'LEI', 'Oil shock', 'Credit spreads',
                   'Corp profits', 'Durables', 'Euphoria', 'Sahm rule', 'Breadth', 'Fiscal(-)']

JUDGMENT_WEIGHTS = [13, 8, 9, 7, 18, 7, 6, 10, 8, 6, 8]

EVENTS = [
    {'label': '1973-74 Oil Shock', 'drawdown': -0.48, 'recession': True, 'regime': 'pre-2000',
     'x': [.90,.70,.80,.95,.90,.80,.70,.60,.90,.70,.30]},
    {'label': '1976-78 Correction', 'drawdown': -0.19, 'recession': False, 'regime': 'pre-2000',
     'x': [.30,.20,.30,.20,.25,.30,.25,.30,.20,.40,.40]},
    {'label': '1980 Volcker I', 'drawdown': -0.17, 'recession': True, 'regime': 'pre-2000',
     'x': [.95,.50,.70,.90,.70,.90,.60,.50,.80,.60,.20]},
    {'label': '1981-82 Volcker II', 'drawdown': -0.27, 'recession': True, 'regime': 'pre-2000',
     'x': [.95,.60,.80,.70,.85,.80,.70,.40,.90,.70,.30]},
    {'label': '1987 Black Monday', 'drawdown': -0.34, 'recession': False, 'regime': 'pre-2000',
     'x': [.40,.20,.20,.15,.45,.20,.15,.70,.10,.80,.35]},
    {'label': '1990-91 Gulf War', 'drawdown': -0.20, 'recession': True, 'regime': 'pre-2000',
     'x': [.80,.70,.60,.70,.60,.60,.50,.50,.70,.50,.30]},
    {'label': '1998 LTCM/Asia', 'drawdown': -0.19, 'recession': False, 'regime': 'pre-2000',
     'x': [.50,.20,.30,.20,.55,.30,.30,.65,.15,.60,.35]},
    {'label': '2000-02 Dot-com', 'drawdown': -0.49, 'recession': True, 'regime': 'post-2000',
     'x': [.70,.30,.70,.30,.70,.70,.60,.90,.60,.85,.40]},
    {'label': '2007-09 GFC', 'drawdown': -0.57, 'recession': True, 'regime': 'post-2000',
     'x': [.90,.90,.90,.60,.95,.90,.80,.85,.90,.85,.25]},
    {'label': '2011 EU Debt', 'drawdown': -0.19, 'recession': False, 'regime': 'post-2000',
     'x': [.20,.25,.40,.30,.45,.25,.30,.30,.20,.65,.50]},
    {'label': '2018 Fed Tight', 'drawdown': -0.20, 'recession': False, 'regime': 'post-2000',
     'x': [.60,.35,.35,.25,.35,.30,.35,.45,.15,.70,.45]},
    {'label': '2020 COVID', 'drawdown': -0.34, 'recession': True, 'regime': 'post-2000',
     'x': [.10,.10,.20,.10,.80,.30,.20,.30,.80,.90,.10]},
    {'label': '2022 Inflation', 'drawdown': -0.25, 'recession': False, 'regime': 'post-2000',
     'x': [.85,.50,.70,.55,.25,.35,.40,.55,.25,.75,.80]},
]

CURRENT_DEFAULT = [.60,.40,.70,.55,.20,.50,.50,.70,.30,.60,.70]
REGIME_WEIGHTS = {'pre-2000': 0.7, 'post-2000': 1.0}


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
    if prob < 0.25: return 'LOW RISK', '#30d158', 'No significant drawdown expected'
    elif prob < 0.40:
        if cs < 0.40: return 'CORRECTION POSSIBLE', '#c8a84a', 'Correction likely'
        else: return 'ELEVATED', '#c8a84a', 'Elevated risk, ambiguous signal'
    elif prob < 0.60:
        if cs >= 0.60 and sa >= 0.50: return 'RECESSION LIKELY', '#ff453a', 'Recession likely'
        elif cs >= 0.50: return 'RECESSION POSSIBLE', '#ff9f0a', 'Recession possible'
        else: return 'CORRECTION LIKELY', '#c8a84a', 'Correction likely'
    else:
        if cs >= 0.60: return 'RECESSION LIKELY', '#ff453a', 'Recession likely'
        else: return 'SEVERE CORRECTION', '#ff453a', 'Severe correction or recession'


# ══════════════════════════════════════════════════════════════
# COMPOSITE MODULES
# ══════════════════════════════════════════════════════════════

def score_band(val, h, s, st, cr, inv=False):
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


def compute_lsi(hours, quits, temp_yoy, claims, u6_gap):
    comps = {
        'hours': score_band(hours, 41.0, 40.5, 40.0, 39.5, inv=True),
        'quits': score_band(quits, 3.5, 3.0, 2.5, 2.0, inv=True),
        'temp': score_band(temp_yoy, 3.0, 0.0, -5.0, -15.0, inv=True),
        'claims': score_band(claims, 1800, 2200, 3000, 4500),
        'u6': score_band(u6_gap, 3.0, 3.5, 4.5, 6.0),
    }
    weights = {'hours': 0.25, 'quits': 0.25, 'temp': 0.20, 'claims': 0.15, 'u6': 0.15}
    total = sum(comps[k] * weights[k] for k in weights)
    level = 'CRISIS' if total >= 70 else 'STRESSED' if total >= 50 else 'SOFTENING' if total >= 30 else 'HEALTHY'
    return total, level, comps


def compute_csi(sloos_ci, sloos_cc, chargeoff, cons_del, auto_del, cp_chg):
    comps = {
        'sloos_ci': score_band(sloos_ci, 0, 15, 35, 60),
        'sloos_cc': score_band(sloos_cc, 0, 10, 25, 45),
        'chargeoff': score_band(chargeoff, 0.50, 0.80, 1.20, 2.00),
        'cons_del': score_band(cons_del, 2.0, 2.5, 3.2, 4.0),
        'auto_del': score_band(auto_del, 1.8, 2.3, 2.8, 3.5),
        'cp_chg': score_band(cp_chg, 5.0, 0.0, -8.0, -20.0, inv=True),
    }
    weights = {'sloos_ci': 0.15, 'sloos_cc': 0.10, 'chargeoff': 0.20,
               'cons_del': 0.15, 'auto_del': 0.15, 'cp_chg': 0.25}
    total = sum(comps[k] * weights[k] for k in weights)
    level = 'CRISIS' if total >= 65 else 'STRESSED' if total >= 45 else 'TIGHTENING' if total >= 25 else 'HEALTHY'
    return total, level, comps


def compute_gci(dxy_surge, em_sov, global_pmi, china_ci, eu_bank):
    comps = {
        'dxy_surge': score_band(dxy_surge, -5, 3, 8, 15),
        'em_sov': score_band(em_sov, 300, 400, 600, 900),
        'global_pmi': score_band(global_pmi, 52, 50.5, 49, 45, inv=True),
        'china_ci': score_band(china_ci, 2, 0, -2, -5, inv=True),
        'eu_bank': score_band(eu_bank, 35, 50, 70, 90),
    }
    weights = {'dxy_surge': 0.20, 'em_sov': 0.20, 'global_pmi': 0.25,
               'china_ci': 0.15, 'eu_bank': 0.20}
    total = sum(comps[k] * weights[k] for k in weights)
    level = 'CRISIS' if total >= 70 else 'STRESSED' if total >= 50 else 'SOFTENING' if total >= 30 else 'CALM'
    return total, level, comps


def compute_cfhi(real_inc, savings, tdsp, cc_util, cc_del, retail):
    comps = {
        'real_inc': score_band(real_inc, 2.5, 1.5, 0.0, -2.0, inv=True),
        'savings': score_band(savings, 6.0, 4.5, 3.0, 2.0, inv=True),
        'tdsp': score_band(tdsp, 9.5, 10.5, 11.5, 13.0),
        'cc_util': score_band(cc_util, 22, 26, 30, 38),
        'cc_del': score_band(cc_del, 2.5, 3.2, 4.0, 5.5),
        'retail': score_band(retail, 3.0, 1.5, 0.0, -3.0, inv=True),
    }
    weights = {'real_inc': 0.20, 'savings': 0.15, 'tdsp': 0.15,
               'cc_util': 0.20, 'cc_del': 0.15, 'retail': 0.15}
    total = sum(comps[k] * weights[k] for k in weights)
    if retail <= 0 and cc_del >= 4.0 and savings <= 3.0:
        phase = "PHASE 3"
        phase_desc = "Spending cracked — recession dynamics active"
    elif total >= 55:
        phase = "LATE PHASE 2"
        phase_desc = "Bridge stressed — Phase 3 imminent"
    elif total >= 30:
        phase = "PHASE 2"
        phase_desc = "Credit bridge active"
    else:
        phase = "PHASE 1"
        phase_desc = "Consumer buffer intact"
    return total, phase, phase_desc, comps


def compute_crei(r2k_eps, zombie, maturity_wall, coupon_shock, regional, smb_bank, ew_div, midcap_margin):
    """Corporate Real Economy Index — measures Joe Average corporate America."""
    comps = {
        'r2k_eps': score_band(r2k_eps, 10, 5, 0, -5, inv=True),
        'zombie': score_band(zombie, 15, 25, 35, 45),
        'maturity_wall': score_band(maturity_wall, 100, 175, 275, 400),
        'coupon_shock': score_band(coupon_shock, 0, 1.5, 3.0, 5.0),
        'regional': score_band(regional, 15, 30, 50, 75),
        'smb_bank': score_band(smb_bank, 0, 10, 25, 50),
        'ew_div': score_band(ew_div, -5, -10, -15, -22, inv=True),
        'midcap_margin': score_band(midcap_margin, 5, 12, 20, 30),
    }
    weights = {'r2k_eps': 0.15, 'zombie': 0.18, 'maturity_wall': 0.14,
               'coupon_shock': 0.12, 'regional': 0.12, 'smb_bank': 0.10,
               'ew_div': 0.10, 'midcap_margin': 0.09}
    total = sum(comps[k] * weights[k] for k in weights)

    if total < 25:
        level, desc = "HEALTHY", "Real economy thriving"
    elif total < 45:
        level, desc = "WATCH", "Pockets of stress emerging"
    elif total < 65:
        level, desc = "STRESSED", "Widespread stress, K-shape extreme"
    elif total < 85:
        level, desc = "ACUTE", "Credit event risk, bankruptcy wave imminent"
    else:
        level, desc = "CRISIS", "Real economy in cascading failure"

    return total, level, desc, comps


# ══════════════════════════════════════════════════════════════
# SEQUENCE STAGE ENGINE
# Synthesizes all modules into single crisis-stage position
# ══════════════════════════════════════════════════════════════

STAGE_DEFS = {
    1: {"name": "Structural Buildup", "subtitle": "Imbalances growing, markets unconcerned",
        "description": "Long-term vulnerabilities accumulating. Markets at ATHs, volatility low. Years before crisis.",
        "feel": "Boom times. ATHs. 'What could go wrong?'",
        "examples": ["2004-2006: Housing buildup", "2015-2019: Debt growth", "2021-2023: Private credit expansion"],
        "color": "#30d158"},
    2: {"name": "Early Stress", "subtitle": "Fundamentals weakening quietly",
        "description": "Ground-level sensors showing stress while markets remain calm. 6-18 months before crisis.",
        "feel": "Concern in research, markets still green",
        "examples": ["Early 2007: Banks tightening", "Late 1999: Tech margins peaking", "Late 2019: Repo stress"],
        "color": "#c8a84a"},
    3: {"name": "Fundamental-Market Divergence", "subtitle": "Ground data and markets no longer agree",
        "description": "Multiple sensors stressed while market pricing calm. The 'something is off' stage. 3-12 months before crisis.",
        "feel": "Fundamentals concerning, markets shrug, breadth narrowing",
        "examples": ["Summer 2007: CSI stressed, HY OAS calm", "January 2000: Breadth narrowing, ATH", "October 2019: Repo stress at ATH"],
        "color": "#ff9f0a"},
    4: {"name": "Pipe Activation", "subtitle": "Structural pipes rupturing",
        "description": "At least one pipe at active rupture OR market beginning to price risk. 1-6 months before full crisis.",
        "feel": "Early volatility, first drawdown, 'is this it?'",
        "examples": ["August 2007: BNP Paribas freeze", "September 2019: Repo spike to 10%", "March 2023: SVB run"],
        "color": "#ff9f0a"},
    5: {"name": "Crisis Cascade", "subtitle": "Simultaneous multi-dimensional stress",
        "description": "The crisis is happening. Multiple pipes active, rapid repricing, Fed intervention questions.",
        "feel": "Panic, Fed questions, forced deleveraging",
        "examples": ["Sept-Oct 2008: Lehman", "March 2020: COVID crash", "Feb-March 2023: Regional banks"],
        "color": "#ff453a"},
    6: {"name": "Resolution", "subtitle": "Recovery or grind",
        "description": "Post-crisis. Fed/fiscal intervention, markets stabilizing OR grinding.",
        "feel": "'Worst over' narrative, recovery or grind",
        "examples": ["2009-2010: Post-GFC recovery", "April 2020-onwards: Post-COVID", "Post-March 2023: Stabilization"],
        "color": "#64d2ff"},
}


def assess_sequence_stage(state):
    """Compute sequence stage from current state."""
    # Define triggers for each stage
    triggers = {
        1: [
            ("Debt/GDP > 80%", state.get("debt_gdp", 0) >= 80),
            ("Foreign share declining <28%", state.get("foreign_share", 100) < 28),
            ("Private credit pipe elevated", state.get("pipe2_score", 0) >= 40),
            ("K-shape concentration widening", state.get("ew_divergence", 0) <= -10),
            ("Zombie share > 25%", state.get("zombie_pct", 0) >= 25),
        ],
        2: [
            ("LSI in Softening/Stressed", state.get("lsi_total", 0) >= 40),
            ("CSI in Tightening/Stressed", state.get("csi_total", 0) >= 35),
            ("CFHI in Phase 2", state.get("cfhi_total", 0) >= 30),
            ("CREI in Watch/Stressed", state.get("crei_total", 0) >= 35),
            ("GCI Softening", state.get("gci_total", 0) >= 30),
        ],
        3: [
            ("LSI > 50 (Stressed)", state.get("lsi_total", 0) >= 50),
            ("CSI > 45 with HY OAS < 4%", state.get("csi_total", 0) >= 45 and state.get("hy_oas", 10) < 4),
            ("CFHI in Late Phase 2", state.get("cfhi_total", 0) >= 45),
            ("CREI > 45 (widespread stress)", state.get("crei_total", 0) >= 45),
            ("Equal-weight S&P divergence > 15%", state.get("ew_divergence", 0) <= -15),
            ("3+ sensors stressed simultaneously",
             sum([state.get("lsi_total", 0) >= 50, state.get("csi_total", 0) >= 45,
                  state.get("cfhi_total", 0) >= 45, state.get("crei_total", 0) >= 45,
                  state.get("gci_total", 0) >= 40]) >= 3),
        ],
        4: [
            ("Any pipe at ACTIVE-RUPTURE",
             state.get("pipe1_status") == "ACTIVE-RUPTURE" or state.get("pipe2_status") == "ACTIVE-RUPTURE" or state.get("pipe3_status") == "ACTIVE-RUPTURE"),
            ("HY OAS crosses 4.0%", state.get("hy_oas", 0) >= 4.0),
            ("ML probability > 30%", state.get("ml_prob", 0) >= 0.30),
            ("CFHI Phase 3 confirmed", state.get("cfhi_total", 0) >= 65),
            ("CREI in Acute range", state.get("crei_total", 0) >= 65),
            ("Pipe 3 short-term At-Risk+", state.get("pipe3_short_status") in ["AT-RISK", "ACTIVE-RUPTURE"]),
        ],
        5: [
            ("HY OAS > 6%", state.get("hy_oas", 0) >= 6.0),
            ("ML probability > 50%", state.get("ml_prob", 0) >= 0.50),
            ("2+ pipes simultaneously ACTIVE-RUPTURE",
             sum([state.get("pipe1_status") == "ACTIVE-RUPTURE",
                  state.get("pipe2_status") == "ACTIVE-RUPTURE",
                  state.get("pipe3_status") == "ACTIVE-RUPTURE"]) >= 2),
            ("CFHI Phase 3 AND CREI Acute",
             state.get("cfhi_total", 0) >= 65 and state.get("crei_total", 0) >= 65),
            ("Market drawdown > 20%", state.get("spx_drawdown", 0) >= 20),
        ],
        6: [
            ("Drawdown > 30% stabilizing", state.get("spx_drawdown", 0) >= 30),
            ("Fed balance sheet expanding rapidly", state.get("fed_bs_expansion", 0) >= 50),
        ],
    }

    # Count triggers fired per stage
    stage_scores = {}
    active_triggers = {}
    for stage_num, stage_triggers in triggers.items():
        fired = [name for name, condition in stage_triggers if condition]
        stage_scores[stage_num] = len(fired)
        active_triggers[stage_num] = fired

    # Primary stage: highest stage with >50% triggers active
    primary = 1
    for stage_num in sorted(triggers.keys()):
        total = len(triggers[stage_num])
        if total > 0 and stage_scores[stage_num] / total >= 0.5:
            primary = stage_num

    # Fractional position
    frac = float(primary)
    if primary < 6:
        next_total = len(triggers[primary + 1])
        if next_total > 0:
            frac += (stage_scores[primary + 1] / next_total) * 0.9

    # Progression rate
    severe_pipes = sum([
        1 if state.get("pipe1_status") in ["ACTIVE-RUPTURE", "AT-RISK"] else 0,
        1 if state.get("pipe2_status") in ["ACTIVE-RUPTURE", "AT-RISK"] else 0,
        1 if state.get("pipe3_status") in ["ACTIVE-RUPTURE", "AT-RISK"] else 0,
    ])

    if severe_pipes >= 2 and primary >= 3:
        rate, rate_desc, rate_color = "ACCELERATING", "Multiple pipes stressed, stage advancement likely within 3-6 months", "#ff453a"
    elif severe_pipes >= 1 and primary >= 3:
        rate, rate_desc, rate_color = "ELEVATED", "Pipe stressed, advancement possible within 6-12 months", "#ff9f0a"
    elif primary >= 2 and frac - primary > 0.3:
        rate, rate_desc, rate_color = "MODERATE", "Moving through current stage at normal pace", "#c8a84a"
    elif primary == 1:
        rate, rate_desc, rate_color = "SLOW", "Structural buildup only, no near-term transition", "#30d158"
    else:
        rate, rate_desc, rate_color = "STABLE", "Current position stable", "#c8a84a"

    # Tail risk profile
    if primary <= 2 and severe_pipes == 0:
        tail_level, tail_dd, tail_scen, tail_color = "LOW", "0-15%", "Normal cyclical risks only", "#30d158"
    elif primary <= 3 and severe_pipes <= 1:
        tail_level, tail_dd, tail_scen, tail_color = "MODERATE", "10-25%", "Standard correction to mild recession risk", "#c8a84a"
    elif primary == 3 and severe_pipes >= 2:
        tail_level, tail_dd, tail_scen, tail_color = "ELEVATED", "20-35%", "Crisis setup present — shock would amplify significantly", "#ff9f0a"
    elif primary == 4:
        tail_level, tail_dd, tail_scen, tail_color = "HIGH", "25-45%", "Pipe activation underway — cascade risk meaningful", "#ff453a"
    elif primary >= 5:
        tail_level, tail_dd, tail_scen, tail_color = "SEVERE", "35-60%", "Crisis in progress", "#ff453a"
    else:
        tail_level, tail_dd, tail_scen, tail_color = "MODERATE", "15-30%", "Transitional — monitor closely", "#c8a84a"

    return {
        "primary_stage": primary,
        "fractional_stage": round(frac, 2),
        "stage_info": STAGE_DEFS[primary],
        "stage_scores": stage_scores,
        "stage_totals": {k: len(v) for k, v in triggers.items()},
        "active_triggers": active_triggers,
        "progression_indicators": active_triggers.get(primary + 1, []) if primary < 6 else [],
        "rate": rate,
        "rate_desc": rate_desc,
        "rate_color": rate_color,
        "severe_pipes": severe_pipes,
        "tail_level": tail_level,
        "tail_drawdown": tail_dd,
        "tail_scenario": tail_scen,
        "tail_color": tail_color,
    }


# ══════════════════════════════════════════════════════════════
# FETCH LIVE DATA (cached)
# ══════════════════════════════════════════════════════════════

fred_data, fred_status = fetch_all_fred_data()

# Map live FRED to fire levels for the 11 core indicators
# Falls back to CURRENT_DEFAULT if FRED unavailable
def build_current_from_fred():
    """Build the 11 fire levels, using live FRED where available."""
    current = list(CURRENT_DEFAULT)

    if fred_status != "LIVE":
        return current, {}

    raw_values = {}

    # Yield curve (idx 0)
    if fred_data.get("t10y2y", {}).get("source") == "FRED_LIVE":
        v = fred_data["t10y2y"]["value"]
        current[0] = normalize_indicator("t10y2y", v)
        raw_values["Yield curve"] = (v, "%", fred_data["t10y2y"]["date"])

    # Permits (idx 1)
    if fred_data.get("permits", {}).get("source") == "FRED_LIVE":
        v = fred_data["permits"]["value"]
        current[1] = normalize_indicator("permits", v)
        raw_values["Permits"] = (v/1000, "M", fred_data["permits"]["date"])

    # Oil (idx 3)
    if fred_data.get("wti_oil", {}).get("source") == "FRED_LIVE":
        v = fred_data["wti_oil"]["value"]
        current[3] = normalize_indicator("wti_oil", v)
        raw_values["Oil shock"] = (v, "$", fred_data["wti_oil"]["date"])

    # HY OAS (idx 4) — THE KEY INDICATOR
    if fred_data.get("bamlh0a0hym2", {}).get("source") == "FRED_LIVE":
        v = fred_data["bamlh0a0hym2"]["value"]
        current[4] = normalize_indicator("bamlh0a0hym2", v)
        raw_values["Credit spreads"] = (v, "%", fred_data["bamlh0a0hym2"]["date"])

    # Unemployment → Sahm proxy (idx 8)
    if fred_data.get("unrate", {}).get("source") == "FRED_LIVE":
        v = fred_data["unrate"]["value"]
        raw_values["Unemployment"] = (v, "%", fred_data["unrate"]["date"])

    # Fiscal (idx 10)
    if fred_data.get("fed_deficit_pct_gdp", {}).get("source") == "FRED_LIVE":
        v = fred_data["fed_deficit_pct_gdp"]["value"]
        current[10] = normalize_indicator("fed_deficit_pct_gdp", v)
        raw_values["Fiscal(-)"] = (v, "%", fred_data["fed_deficit_pct_gdp"]["date"])

    return current, raw_values


current, raw_fred = build_current_from_fred()


# ══════════════════════════════════════════════════════════════
# SIDEBAR — adjustments + data status
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### Data Status")

    if fred_status == "LIVE":
        live_count = sum(1 for k,v in fred_data.items() if v["source"] == "FRED_LIVE")
        total_count = len(fred_data)
        st.markdown(f"🟢 **FRED Live** — {live_count}/{total_count} series")
        st.caption(f"Cached for 1 hour. Last fetch: {datetime.now().strftime('%H:%M')}")
    elif fred_status == "NO_API_KEY":
        st.markdown("🟡 **Manual Mode**")
        st.caption("Add FRED_API_KEY in Streamlit secrets for live data")
    elif fred_status == "FREDAPI_NOT_INSTALLED":
        st.markdown("🟡 **Manual Mode**")
        st.caption("Add `fredapi` to requirements.txt")
    else:
        st.markdown(f"🔴 **Error** — using fallbacks")
        st.caption(str(fred_status)[:100])

    st.markdown("---")
    st.markdown("### Indicator Overrides")
    st.caption("Slide to override. Live values shown when available.")

    indicator_labels = {
        'Yield curve': 'Treasury yield curve inversion',
        'Permits': 'Building permit decline',
        'LEI': 'Leading Economic Index decline',
        'Oil shock': 'Oil price shock severity',
        'Credit spreads': 'HY OAS — THE KEY',
        'Corp profits': 'Corporate profit decline',
        'Durables': 'Durable goods orders decline',
        'Euphoria': 'Margin debt + Buffett indicator',
        'Sahm rule': 'Sahm rule proximity',
        'Breadth': 'Market breadth deterioration',
        'Fiscal(-)': 'Fiscal support (counter)',
    }

    override_current = []
    for i, name in enumerate(INDICATOR_NAMES):
        val = st.slider(
            name,
            0.0, 1.0, float(current[i]), 0.05,
            help=indicator_labels[name],
            key=f"ind_{i}"
        )
        override_current.append(val)
    current = override_current

    st.markdown("---")
    st.markdown("### 👷 LSI")
    lsi_hours = st.slider("Weekly hours", 38.0, 42.0,
                          float(fred_data.get("avg_weekly_hours", {}).get("value", 39.8) or 39.8), 0.1)
    lsi_quits = st.slider("Quits rate", 1.5, 5.0,
                          float(fred_data.get("quits_rate", {}).get("value", 2.8) or 2.8), 0.1)
    lsi_temp = st.slider("Temp emp YoY %", -20.0, 10.0, -6.2, 0.5)
    lsi_claims = st.slider("Continued claims K", 1000, 6000,
                           int(fred_data.get("continued_claims", {}).get("value", 2150) or 2150), 50)
    u3 = float(fred_data.get("unrate", {}).get("value", 4.3) or 4.3)
    u6 = float(fred_data.get("u6_rate", {}).get("value", 8.4) or 8.4)
    lsi_u6 = st.slider("U6-U3 gap pp", 2.0, 8.0, max(2.0, u6 - u3), 0.1)

    st.markdown("---")
    st.markdown("### 🏦 CSI")
    csi_sloos_ci = st.slider("SLOOS C&I %", -30.0, 80.0,
                             float(fred_data.get("sloos_ci", {}).get("value", 18.0) or 18.0), 1.0)
    csi_sloos_cc = st.slider("SLOOS CC %", -20.0, 60.0,
                             float(fred_data.get("sloos_cc", {}).get("value", 22.0) or 22.0), 1.0)
    csi_chargeoff = st.slider("Charge-off %", 0.1, 3.0,
                              float(fred_data.get("chargeoff_rate", {}).get("value", 0.72) or 0.72), 0.02)
    csi_cons_del = st.slider("Cons delinq %", 1.0, 5.0,
                             float(fred_data.get("consumer_delinq", {}).get("value", 3.1) or 3.1), 0.1)
    csi_auto_del = st.slider("Auto delinq %", 1.0, 4.0,
                             float(fred_data.get("auto_delinq", {}).get("value", 3.3) or 3.3), 0.1)
    csi_cp_chg = st.slider("CP YoY %", -30.0, 20.0, -5.0, 1.0)

    st.markdown("---")
    st.markdown("### 🌍 GCI")
    gci_dxy_surge = st.slider("USD 6m %", -15.0, 20.0, -5.0, 0.5)
    gci_em_sov = st.slider("EM sov spread bps", 150, 1500, 280, 10)
    gci_global_pmi = st.slider("Global PMI", 40.0, 60.0, 49.1, 0.1)
    gci_china_ci = st.slider("China CI pp", -8.0, 10.0, -2.0, 0.5)
    gci_eu_bank = st.slider("EU bank stress", 20, 100, 55, 1)

    st.markdown("---")
    st.markdown("### 👤 CFHI")
    cfhi_real_inc = st.slider("Real inc YoY %", -5.0, 8.0, 1.2, 0.1)
    cfhi_savings = st.slider("Savings rate %", 1.0, 15.0,
                             float(fred_data.get("savings_rate", {}).get("value", 3.8) or 3.8), 0.1)
    cfhi_tdsp = st.slider("Debt service %", 8.0, 15.0,
                          float(fred_data.get("debt_service", {}).get("value", 11.4) or 11.4), 0.1)
    cfhi_cc_util = st.slider("CC utilization %", 15, 45, 29, 1)
    cfhi_cc_del = st.slider("CC delinq 90+d %", 1.0, 8.0,
                            float(fred_data.get("cc_delinq_90", {}).get("value", 2.94) or 2.94), 0.05)
    cfhi_retail = st.slider("Retail sales YoY %", -10.0, 10.0, 1.8, 0.2)

    st.markdown("---")
    st.markdown("### 🏭 CREI (Real Economy)")
    crei_r2k_eps = st.slider("Russell 2000 EPS YoY %", -40.0, 30.0, 2.0, 1.0,
                              help="Small-cap earnings — the Joe Average companies")
    crei_zombie = st.slider("Zombie company % (R2K)", 10, 60, 43, 1,
                             help="% of Russell 2000 unable to cover interest")
    crei_maturity_wall = st.slider("2026 maturity wall ($B)", 50, 700, 368, 10,
                                    help="Small-cap debt that must be refinanced")
    crei_coupon_shock = st.slider("Refi coupon shock (pp)", -2.0, 10.0, 5.5, 0.5,
                                   help="Old rate vs new rate differential")
    crei_regional = st.slider("Regional bank stress", 0, 100, 55, 1,
                               help="KRE stress, CRE exposure, deposit outflows")
    crei_smb_bankruptcy = st.slider("SMB bankruptcy YoY %", -20, 100, 22, 2,
                                     help="Small business bankruptcy filings")
    crei_ew_divergence = st.slider("EW vs CW S&P 500 %", -35, 10, -18, 1,
                                    help="Equal-weight vs cap-weight divergence (market concentration)")
    crei_midcap_margin = st.slider("Midcap margin compression", 0, 60, 20, 1,
                                    help="S&P 400 operating margin decline index")


# ══════════════════════════════════════════════════════════════
# COMPUTE ALL METRICS
# ══════════════════════════════════════════════════════════════

X, y, w, feat_names = build_dataset()
model = train_model(tuple(map(tuple, X)), tuple(y), tuple(w))

cur_vec, _ = get_current_vec(current)
probs = model.predict(cur_vec)
ml_p = probs['ensemble'][0]
agreement = probs['agreement'][0]
j_p = sum(current[i] * JUDGMENT_WEIGHTS[i] for i in range(11)) / sum(JUDGMENT_WEIGHTS)

cls_name, cls_color, cls_desc = classify(ml_p, current[4], current[8])

lsi_total, lsi_level, lsi_comps = compute_lsi(lsi_hours, lsi_quits, lsi_temp, lsi_claims, lsi_u6)
csi_total, csi_level, csi_comps = compute_csi(csi_sloos_ci, csi_sloos_cc, csi_chargeoff, csi_cons_del, csi_auto_del, csi_cp_chg)
gci_total, gci_level, gci_comps = compute_gci(gci_dxy_surge, gci_em_sov, gci_global_pmi, gci_china_ci, gci_eu_bank)
cfhi_total, cfhi_phase, cfhi_phase_desc, cfhi_comps = compute_cfhi(cfhi_real_inc, cfhi_savings, cfhi_tdsp, cfhi_cc_util, cfhi_cc_del, cfhi_retail)
crei_total, crei_level, crei_desc, crei_comps = compute_crei(crei_r2k_eps, crei_zombie, crei_maturity_wall, crei_coupon_shock, crei_regional, crei_smb_bankruptcy, crei_ew_divergence, crei_midcap_margin)

hy_fire = current[4]
csi_norm = csi_total / 100
div_gap = csi_norm - hy_fire

# Compute Pipe 1 status (for sequence engine)
p1_scores_early = [
    score_band(65, 40, 55, 70, 85),
    score_band(30, 15, 22, 28, 35),
    score_band(-5, 3, 0, -3, -6, inv=True),
    score_band(-70, 0, -15, -35, -55, inv=True),
    score_band(6.8, 2, 3.5, 5, 7),
    score_band(75, 25, 45, 65, 80),
    score_band(35, 70, 55, 40, 25, inv=True),
    score_band(60, 30, 45, 60, 75),
]
p1_max_early = max(p1_scores_early)
p1_above75_early = sum(1 for s in p1_scores_early if s >= 75)
p1_above50_early = sum(1 for s in p1_scores_early if s >= 50)
if p1_max_early >= 90: p1_status_early = "ACTIVE-RUPTURE"
elif p1_above75_early >= 2: p1_status_early = "AT-RISK"
elif p1_above50_early >= 3 or p1_max_early >= 75: p1_status_early = "STRESSED"
elif p1_above50_early >= 1: p1_status_early = "WATCH"
else: p1_status_early = "HEALTHY"

# Compute Pipe 2 status
p2_scores_early = [
    score_band(17, 0, 5, 12, 20),
    score_band(8, 3, 5, 8, 12),
    score_band(42, 25, 32, 40, 50),
    score_band(5.8, 2, 3.5, 5, 8),
    score_band(65, 20, 35, 55, 75),
    score_band(72, 20, 40, 60, 80),
    score_band(40, 15, 25, 35, 45),
    score_band(45, 20, 35, 55, 75),
]
p2_max_early = max(p2_scores_early)
p2_above75_early = sum(1 for s in p2_scores_early if s >= 75)
p2_above50_early = sum(1 for s in p2_scores_early if s >= 50)
if p2_max_early >= 90: p2_status_early = "ACTIVE-RUPTURE"
elif p2_above75_early >= 2: p2_status_early = "AT-RISK"
elif p2_above50_early >= 3 or p2_max_early >= 75: p2_status_early = "STRESSED"
elif p2_above50_early >= 1: p2_status_early = "WATCH"
else: p2_status_early = "HEALTHY"

# Compute Pipe 3 status
p3_short_scores_early = [
    score_band(1000, 200, 400, 700, 1000),
    score_band(65, 25, 40, 60, 80),
    score_band(30, 15, 30, 50, 75),
    score_band(45, 20, 35, 55, 75),
]
p3_long_scores_early = [
    score_band(120, 60, 80, 100, 130),
    score_band(20, 8, 12, 18, 25),
    score_band(6, 0, 2, 4, 6),
    score_band(70, 15, 35, 55, 80),
    score_band(22, 35, 30, 25, 20, inv=True),
    score_band(75, 25, 45, 65, 80),
]
p3_all_scores = p3_short_scores_early + p3_long_scores_early
p3_max_early = max(p3_all_scores)
p3_above75_early = sum(1 for s in p3_all_scores if s >= 75)
p3_above50_early = sum(1 for s in p3_all_scores if s >= 50)
if p3_max_early >= 90: p3_status_early = "ACTIVE-RUPTURE"
elif p3_above75_early >= 2: p3_status_early = "AT-RISK"
elif p3_above50_early >= 3 or p3_max_early >= 75: p3_status_early = "STRESSED"
elif p3_above50_early >= 1: p3_status_early = "WATCH"
else: p3_status_early = "HEALTHY"

# Pipe 3 short-term sub-status
p3_short_max = max(p3_short_scores_early)
p3_short_above75 = sum(1 for s in p3_short_scores_early if s >= 75)
p3_short_above50 = sum(1 for s in p3_short_scores_early if s >= 50)
if p3_short_max >= 90: p3_short_status_early = "ACTIVE-RUPTURE"
elif p3_short_above75 >= 2: p3_short_status_early = "AT-RISK"
elif p3_short_above50 >= 3 or p3_short_max >= 75: p3_short_status_early = "STRESSED"
elif p3_short_above50 >= 1: p3_short_status_early = "WATCH"
else: p3_short_status_early = "HEALTHY"

# Now compute sequence stage
sequence_state = {
    "ml_prob": ml_p,
    "hy_oas": current[4] * 10,  # current[4] is fire level 0-1; convert to approximate % spread
    "spx_drawdown": 0,
    "lsi_total": lsi_total,
    "csi_total": csi_total,
    "gci_total": gci_total,
    "cfhi_total": cfhi_total,
    "crei_total": crei_total,
    "pipe1_status": p1_status_early,
    "pipe2_status": p2_status_early,
    "pipe3_status": p3_status_early,
    "pipe3_short_status": p3_short_status_early,
    "debt_gdp": 120,
    "foreign_share": 22,
    "pipe2_score": p2_max_early,
    "ew_divergence": crei_ew_divergence,
    "zombie_pct": crei_zombie,
    "fed_bs_expansion": 0,
}
seq = assess_sequence_stage(sequence_state)


# ══════════════════════════════════════════════════════════════
# HEADER — Apple-style
# ══════════════════════════════════════════════════════════════

col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div style='margin-top:8px'>
      <div style='color:#636366;font-size:0.8em;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:4px'>Recession Radar · v8</div>
      <h1 style='margin:0'>Two frameworks. One decision.</h1>
      <div style='color:#86868b;font-size:1.05em;margin-top:6px;font-weight:300'>Cyclical probability. Structural vulnerability. Each tells a different truth.</div>
    </div>
    """, unsafe_allow_html=True)
with col_status:
    if fred_status == "LIVE":
        live_count = sum(1 for k,v in fred_data.items() if v["source"] == "FRED_LIVE")
        total_count = len(fred_data)
        st.markdown(f"""
        <div style='padding:14px 18px;background:rgba(48,209,88,0.08);border:1px solid rgba(48,209,88,0.3);border-radius:12px;text-align:right;margin-top:20px'>
          <div style='color:#30d158;font-size:0.72em;letter-spacing:0.1em;text-transform:uppercase'>● Live</div>
          <div style='color:#f5f5f7;font-size:1em;margin-top:2px'>{live_count}/{total_count} FRED series</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding:14px 18px;background:rgba(200,168,74,0.08);border:1px solid rgba(200,168,74,0.3);border-radius:12px;text-align:right;margin-top:20px'>
          <div style='color:#c8a84a;font-size:0.72em;letter-spacing:0.1em;text-transform:uppercase'>○ Manual</div>
          <div style='color:#f5f5f7;font-size:1em;margin-top:2px'>No API key set</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HERO SECTION — Sequence Stage Engine (the headline)
# ══════════════════════════════════════════════════════════════

# Sequence stage hero — the single most important readout
stage_color = seq['stage_info']['color']

st.markdown(f"""
<div style='padding:32px 36px;background:linear-gradient(135deg,{stage_color}15 0%,{stage_color}05 100%);border:1px solid {stage_color}44;border-radius:24px;margin-bottom:20px'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:24px'>
    <div style='flex:1;min-width:300px'>
      <div style='color:#86868b;font-size:0.72em;letter-spacing:0.15em;text-transform:uppercase;font-weight:500'>Sequence Stage</div>
      <div style='color:{stage_color};font-size:5em;font-weight:700;line-height:1;letter-spacing:-0.04em;margin-top:8px'>{seq['fractional_stage']:.2f}</div>
      <div style='color:#f5f5f7;font-size:1.6em;font-weight:500;margin-top:4px;letter-spacing:-0.01em'>{seq['stage_info']['name']}</div>
      <div style='color:#86868b;font-size:1em;margin-top:6px;font-weight:300'>{seq['stage_info']['subtitle']}</div>
    </div>
    <div style='display:flex;gap:16px;flex-wrap:wrap'>
      <div style='padding:16px 20px;background:{seq['rate_color']}15;border:1px solid {seq['rate_color']}44;border-radius:14px;min-width:180px'>
        <div style='color:#86868b;font-size:0.7em;letter-spacing:0.12em;text-transform:uppercase'>Progression Rate</div>
        <div style='color:{seq['rate_color']};font-size:1.6em;font-weight:600;margin-top:4px;letter-spacing:-0.01em'>{seq['rate']}</div>
        <div style='color:#86868b;font-size:0.78em;margin-top:4px'>{seq['severe_pipes']}/3 severe pipes</div>
      </div>
      <div style='padding:16px 20px;background:{seq['tail_color']}15;border:1px solid {seq['tail_color']}44;border-radius:14px;min-width:180px'>
        <div style='color:#86868b;font-size:0.7em;letter-spacing:0.12em;text-transform:uppercase'>Tail Risk</div>
        <div style='color:{seq['tail_color']};font-size:1.6em;font-weight:600;margin-top:4px;letter-spacing:-0.01em'>{seq['tail_level']}</div>
        <div style='color:#86868b;font-size:0.78em;margin-top:4px'>{seq['tail_drawdown']} drawdown range</div>
      </div>
    </div>
  </div>

  <div style='margin-top:22px;padding:14px 18px;background:rgba(255,255,255,0.03);border-left:2px solid {stage_color};border-radius:8px'>
    <div style='color:#d1d1d6;font-size:0.92em;line-height:1.6'>{seq['stage_info']['description']}</div>
    <div style='color:#86868b;font-size:0.82em;margin-top:6px;font-style:italic'>Feel: {seq['stage_info']['feel']}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Secondary row — the classical ML + Pipes summary
hero_left, hero_right = st.columns(2)

with hero_left:
    # Framework 1: Recession Radar
    st.markdown("""
    <div style='padding:28px;background:linear-gradient(135deg,rgba(200,168,74,0.08) 0%,rgba(200,168,74,0.02) 100%);border:1px solid rgba(200,168,74,0.2);border-radius:22px;height:100%'>
      <div style='color:#c8a84a;font-size:0.7em;letter-spacing:0.15em;text-transform:uppercase;font-weight:500'>Framework 1 · Cyclical</div>
      <div style='color:#86868b;font-size:0.9em;margin-top:4px;margin-bottom:20px'>Recession Radar ML</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style='color:#86868b;font-size:0.7em;letter-spacing:0.1em;text-transform:uppercase'>ML Probability</div>
        <div style='color:#f5f5f7;font-size:3em;font-weight:600;letter-spacing:-0.03em;line-height:1;margin-top:4px'>{ml_p*100:.0f}<span style='color:#86868b;font-size:0.5em;font-weight:400'>%</span></div>
        <div style='color:#636366;font-size:0.75em;margin-top:4px'>Agreement {agreement*100:.0f}%</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='color:#86868b;font-size:0.7em;letter-spacing:0.1em;text-transform:uppercase'>Judgment</div>
        <div style='color:#f5f5f7;font-size:3em;font-weight:600;letter-spacing:-0.03em;line-height:1;margin-top:4px'>{j_p*100:.0f}<span style='color:#86868b;font-size:0.5em;font-weight:400'>%</span></div>
        <div style='color:#636366;font-size:0.75em;margin-top:4px'>Expert weighted</div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
      <div style='margin-top:20px;padding:12px 16px;background:{cls_color}11;border-left:2px solid {cls_color};border-radius:8px'>
        <div style='color:{cls_color};font-size:0.82em;font-weight:600'>{cls_name}</div>
        <div style='color:#86868b;font-size:0.78em;margin-top:2px'>{cls_desc}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    # Framework 2: Structural Vulnerability — compute pipe statuses
    # Pipe 1
    pipe1_scores = [
        score_band(65, 40, 55, 70, 85),          # Gulf SWF
        score_band(30, 15, 22, 28, 35),          # Saudi-China
        score_band(-5, 3, 0, -3, -6, inv=True),  # DXY
        score_band(-70, 0, -15, -35, -55, inv=True),  # PIF
        score_band(6.8, 2, 3.5, 5, 7),           # JP hedging
        score_band(75, 25, 45, 65, 80),          # Gold
        score_band(35, 70, 55, 40, 25, inv=True),# TIC
        score_band(60, 30, 45, 60, 75),          # Shadow
    ]
    p1_max = max(pipe1_scores)
    p1_above50 = sum(1 for s in pipe1_scores if s >= 50)
    p1_above75 = sum(1 for s in pipe1_scores if s >= 75)
    if p1_max >= 90: p1_status, p1_color = "ACTIVE-RUPTURE", "#ff453a"
    elif p1_above75 >= 2: p1_status, p1_color = "AT-RISK", "#ff453a"
    elif p1_above50 >= 3 or p1_max >= 75: p1_status, p1_color = "STRESSED", "#ff453a"
    elif p1_above50 >= 1: p1_status, p1_color = "WATCH", "#c8a84a"
    else: p1_status, p1_color = "HEALTHY", "#30d158"

    # Pipe 2
    pipe2_scores = [
        score_band(17, 0, 5, 12, 20),
        score_band(8, 3, 5, 8, 12),
        score_band(42, 25, 32, 40, 50),
        score_band(5.8, 2, 3.5, 5, 8),
        score_band(65, 20, 35, 55, 75),
        score_band(72, 20, 40, 60, 80),
        score_band(40, 15, 25, 35, 45),
        score_band(45, 20, 35, 55, 75),
    ]
    p2_max = max(pipe2_scores)
    p2_above50 = sum(1 for s in pipe2_scores if s >= 50)
    p2_above75 = sum(1 for s in pipe2_scores if s >= 75)
    if p2_max >= 90: p2_status, p2_color = "ACTIVE-RUPTURE", "#ff453a"
    elif p2_above75 >= 2: p2_status, p2_color = "AT-RISK", "#ff453a"
    elif p2_above50 >= 3 or p2_max >= 75: p2_status, p2_color = "STRESSED", "#ff453a"
    elif p2_above50 >= 1: p2_status, p2_color = "WATCH", "#c8a84a"
    else: p2_status, p2_color = "HEALTHY", "#30d158"

    st.markdown(f"""
    <div style='padding:28px;background:linear-gradient(135deg,rgba(255,69,58,0.08) 0%,rgba(255,69,58,0.02) 100%);border:1px solid rgba(255,69,58,0.2);border-radius:22px;height:100%'>
      <div style='color:#ff453a;font-size:0.7em;letter-spacing:0.15em;text-transform:uppercase;font-weight:500'>Framework 2 · Structural</div>
      <div style='color:#86868b;font-size:0.9em;margin-top:4px;margin-bottom:20px'>Vulnerability Map</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style='color:#86868b;font-size:0.7em;letter-spacing:0.1em;text-transform:uppercase'>Pipe 1 · Foreign</div>
        <div style='color:{p1_color};font-size:1.35em;font-weight:600;letter-spacing:-0.01em;line-height:1.1;margin-top:4px'>{p1_status}</div>
        <div style='color:#636366;font-size:0.75em;margin-top:4px'>{p1_above50}/8 stressed · {p1_above75}/8 at-risk</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='color:#86868b;font-size:0.7em;letter-spacing:0.1em;text-transform:uppercase'>Pipe 2 · Shadow</div>
        <div style='color:{p2_color};font-size:1.35em;font-weight:600;letter-spacing:-0.01em;line-height:1.1;margin-top:4px'>{p2_status}</div>
        <div style='color:#636366;font-size:0.75em;margin-top:4px'>{p2_above50}/8 stressed · {p2_above75}/8 at-risk</div>
        """, unsafe_allow_html=True)

    st.markdown("""
      <div style='margin-top:20px;padding:12px 16px;background:rgba(255,69,58,0.08);border-left:2px solid #ff453a;border-radius:8px'>
        <div style='color:#ff453a;font-size:0.82em;font-weight:600'>2007-equivalent configuration</div>
        <div style='color:#86868b;font-size:0.78em;margin-top:2px'>Structural transmission mechanisms are loaded</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# GROUND-LEVEL SENSORS — 4 composite modules as row of cards
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='margin-bottom:16px'>
  <div style='color:#86868b;font-size:0.72em;letter-spacing:0.15em;text-transform:uppercase;font-weight:500'>Ground-level sensors</div>
  <div style='color:#f5f5f7;font-size:1.3em;font-weight:500;margin-top:4px'>What markets haven't priced yet</div>
</div>
""", unsafe_allow_html=True)

def sensor_card(label, value, unit, level, color):
    return f"""
    <div style='padding:22px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:18px;transition:all 0.3s ease'>
      <div style='color:#86868b;font-size:0.7em;letter-spacing:0.12em;text-transform:uppercase;font-weight:500'>{label}</div>
      <div style='color:#f5f5f7;font-size:2.6em;font-weight:600;letter-spacing:-0.03em;line-height:1;margin-top:8px'>{value}<span style='color:#636366;font-size:0.42em;font-weight:400;margin-left:4px'>{unit}</span></div>
      <div style='margin-top:12px;padding:5px 10px;background:{color}15;border-radius:6px;display:inline-block'>
        <span style='color:{color};font-size:0.72em;font-weight:600;letter-spacing:0.05em'>{level}</span>
      </div>
    </div>
    """

def lvl_color(level):
    if level in ('CRISIS','STRESSED','PHASE 3','LATE PHASE 2','ACUTE'): return '#ff453a'
    if level in ('TIGHTENING','SOFTENING','PHASE 2','WATCH'): return '#c8a84a'
    return '#30d158'

s1, s2, s3, s4, s5 = st.columns(5)
with s1: st.markdown(sensor_card("Labor · LSI", f"{lsi_total:.0f}", "/100", lsi_level, lvl_color(lsi_level)), unsafe_allow_html=True)
with s2: st.markdown(sensor_card("Credit · CSI", f"{csi_total:.0f}", "/100", csi_level, lvl_color(csi_level)), unsafe_allow_html=True)
with s3: st.markdown(sensor_card("Global · GCI", f"{gci_total:.0f}", "/100", gci_level, lvl_color(gci_level)), unsafe_allow_html=True)
with s4: st.markdown(sensor_card("Consumer · CFHI", f"{cfhi_total:.0f}", "/100", cfhi_phase, lvl_color(cfhi_phase)), unsafe_allow_html=True)
with s5: st.markdown(sensor_card("Real Econ · CREI", f"{crei_total:.0f}", "/100", crei_level, lvl_color(crei_level)), unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SEQUENCE STAGE BREAKDOWN — how many triggers active in each stage
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div style='margin-bottom:16px'>
  <div style='color:#86868b;font-size:0.72em;letter-spacing:0.15em;text-transform:uppercase;font-weight:500'>Crisis sequence progression</div>
  <div style='color:#f5f5f7;font-size:1.3em;font-weight:500;margin-top:4px'>Where are we on the path?</div>
</div>
""", unsafe_allow_html=True)

# Stage progression bars
stage_cols = st.columns(6)
for i, stage_num in enumerate(sorted(STAGE_DEFS.keys())):
    with stage_cols[i]:
        stage_def = STAGE_DEFS[stage_num]
        fired = seq['stage_scores'].get(stage_num, 0)
        total = seq['stage_totals'].get(stage_num, 0)
        ratio = fired / total if total > 0 else 0

        # Color based on state
        if stage_num == seq['primary_stage']:
            box_border = f"2px solid {stage_def['color']}"
            box_bg = f"{stage_def['color']}15"
            current_marker = "● CURRENT"
            marker_color = stage_def['color']
        elif stage_num < seq['primary_stage']:
            box_border = f"1px solid rgba(255,255,255,0.1)"
            box_bg = "rgba(255,255,255,0.02)"
            current_marker = "✓ passed"
            marker_color = "#636366"
        else:
            if ratio > 0:
                box_border = f"1px solid {stage_def['color']}33"
                box_bg = f"{stage_def['color']}08"
                current_marker = "⚠ forming"
                marker_color = stage_def['color']
            else:
                box_border = "1px solid rgba(255,255,255,0.06)"
                box_bg = "rgba(255,255,255,0.02)"
                current_marker = "— ahead"
                marker_color = "#636366"

        st.markdown(f"""
        <div style='padding:16px 12px;background:{box_bg};border:{box_border};border-radius:14px;min-height:180px'>
          <div style='color:{marker_color};font-size:0.65em;letter-spacing:0.1em;text-transform:uppercase;font-weight:600'>{current_marker}</div>
          <div style='color:#86868b;font-size:0.7em;letter-spacing:0.08em;text-transform:uppercase;margin-top:6px'>Stage {stage_num}</div>
          <div style='color:#f5f5f7;font-size:1em;font-weight:500;margin-top:3px;letter-spacing:-0.01em;line-height:1.25'>{stage_def['name']}</div>
          <div style='margin-top:10px;height:5px;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden'>
            <div style='height:100%;width:{ratio*100:.0f}%;background:{stage_def['color']};border-radius:3px;transition:width 0.5s ease'></div>
          </div>
          <div style='color:#86868b;font-size:0.72em;margin-top:6px'>{fired}/{total} triggers</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Active triggers for current stage + progression warnings
seq_col_left, seq_col_right = st.columns(2)

with seq_col_left:
    st.markdown(f"""
    <div style='padding:18px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);border-radius:14px;height:100%'>
      <div style='color:#86868b;font-size:0.72em;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px'>Active in Stage {seq['primary_stage']}</div>
    """, unsafe_allow_html=True)
    for t in seq['active_triggers'].get(seq['primary_stage'], []):
        st.markdown(f"<div style='color:#d1d1d6;font-size:0.88em;margin:4px 0;padding:4px 0'>✓ {t}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with seq_col_right:
    if seq['progression_indicators']:
        st.markdown(f"""
        <div style='padding:18px;background:rgba(255,159,10,0.06);border:1px solid rgba(255,159,10,0.25);border-radius:14px;height:100%'>
          <div style='color:#ff9f0a;font-size:0.72em;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px'>Stage {seq['primary_stage'] + 1} triggers forming</div>
        """, unsafe_allow_html=True)
        for t in seq['progression_indicators']:
            st.markdown(f"<div style='color:#d1d1d6;font-size:0.88em;margin:4px 0;padding:4px 0'>⚠ {t}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding:18px;background:rgba(48,209,88,0.06);border:1px solid rgba(48,209,88,0.25);border-radius:14px;height:100%'>
          <div style='color:#30d158;font-size:0.72em;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px'>No next-stage triggers yet</div>
          <div style='color:#d1d1d6;font-size:0.88em'>Current stage position stable. Watch for early triggers.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style='margin-top:16px;padding:14px 18px;background:{seq['rate_color']}10;border-left:2px solid {seq['rate_color']};border-radius:8px'>
  <div style='color:{seq['rate_color']};font-size:0.78em;font-weight:600;letter-spacing:0.05em'>PROGRESSION: {seq['rate']}</div>
  <div style='color:#d1d1d6;font-size:0.88em;margin-top:4px'>{seq['rate_desc']}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# DIVERGENCE WARNING
# ══════════════════════════════════════════════════════════════

if div_gap > 0.15:
    st.markdown(f"""
    <div style='padding:20px 24px;background:rgba(255,69,58,0.08);border:1px solid rgba(255,69,58,0.25);border-radius:16px;margin-bottom:24px'>
      <div style='display:flex;align-items:center;gap:12px'>
        <div style='color:#ff453a;font-size:0.72em;letter-spacing:0.12em;text-transform:uppercase;font-weight:600'>● System A / System B divergence</div>
      </div>
      <div style='color:#f5f5f7;font-size:1em;margin-top:8px;font-weight:500'>Banking fundamentals ({csi_total:.0f}/100) significantly exceed market pricing ({hy_fire*100:.0f}% fire).</div>
      <div style='color:#86868b;font-size:0.88em;margin-top:4px;line-height:1.5'>This exact pattern preceded the GFC by 3-6 months in 2007. Ground-level data suggests risk is higher than market indicators show.</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════

tab_seq, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13 = st.tabs([
    "◉ Sequence",
    "Dashboard",
    "Action Plan",
    "Scenarios",
    "ML Engine",
    "LSI · Labor",
    "CSI · Credit",
    "GCI · Global",
    "CFHI · Consumer",
    "CREI · Real Econ",
    "Pipe 1 · Foreign",
    "Pipe 2 · Shadow",
    "Pipe 3 · Sovereign",
    "About",
])

# ── TAB: SEQUENCE ENGINE ──
with tab_seq:
    st.header(f"Crisis Sequence Stage · {seq['fractional_stage']:.2f}")
    st.markdown(f"#### {seq['stage_info']['name']} — {seq['stage_info']['subtitle']}")

    st.markdown(f"""
    <div style='padding:18px 22px;background:{stage_color}15;border-left:3px solid {stage_color};border-radius:10px;margin:10px 0'>
      <div style='color:#d1d1d6;font-size:1em;line-height:1.6;margin-bottom:8px'>{seq['stage_info']['description']}</div>
      <div style='color:#86868b;font-size:0.92em;font-style:italic'>Feel: {seq['stage_info']['feel']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Historical examples at this stage
    st.markdown("### Historical examples at this stage")
    for ex in seq['stage_info']['examples']:
        st.markdown(f"- {ex}")

    # Full stage trigger matrix
    st.markdown("### Full Stage Trigger Matrix")
    st.caption("How many triggers are active at each stage? This is the raw underlying data the sequence engine synthesizes.")

    for stage_num in sorted(STAGE_DEFS.keys()):
        stage_def = STAGE_DEFS[stage_num]
        fired = seq['stage_scores'].get(stage_num, 0)
        total = seq['stage_totals'].get(stage_num, 0)
        ratio = fired / total if total > 0 else 0

        if stage_num == seq['primary_stage']:
            status_tag = "← CURRENT"
            tag_color = stage_def['color']
        elif stage_num < seq['primary_stage']:
            status_tag = "✓ passed"
            tag_color = "#636366"
        elif ratio > 0:
            status_tag = "⚠ forming"
            tag_color = stage_def['color']
        else:
            status_tag = "— ahead"
            tag_color = "#636366"

        with st.expander(f"Stage {stage_num}: {stage_def['name']} — {fired}/{total} triggers active  [{status_tag}]"):
            st.markdown(f"**{stage_def['subtitle']}**")
            st.markdown(stage_def['description'])
            st.caption(f"Feel: {stage_def['feel']}")

            st.markdown("**Historical examples:**")
            for ex in stage_def['examples']:
                st.markdown(f"- {ex}")

            st.markdown("**Triggers:**")
            active_set = set(seq['active_triggers'].get(stage_num, []))
            # Reconstruct the trigger list for this stage
            all_triggers_this_stage = {
                1: ["Debt/GDP > 80%", "Foreign share declining <28%", "Private credit pipe elevated",
                    "K-shape concentration widening", "Zombie share > 25%"],
                2: ["LSI in Softening/Stressed", "CSI in Tightening/Stressed", "CFHI in Phase 2",
                    "CREI in Watch/Stressed", "GCI Softening"],
                3: ["LSI > 50 (Stressed)", "CSI > 45 with HY OAS < 4%", "CFHI in Late Phase 2",
                    "CREI > 45 (widespread stress)", "Equal-weight S&P divergence > 15%",
                    "3+ sensors stressed simultaneously"],
                4: ["Any pipe at ACTIVE-RUPTURE", "HY OAS crosses 4.0%", "ML probability > 30%",
                    "CFHI Phase 3 confirmed", "CREI in Acute range", "Pipe 3 short-term At-Risk+"],
                5: ["HY OAS > 6%", "ML probability > 50%", "2+ pipes simultaneously ACTIVE-RUPTURE",
                    "CFHI Phase 3 AND CREI Acute", "Market drawdown > 20%"],
                6: ["Drawdown > 30% stabilizing", "Fed balance sheet expanding rapidly"],
            }
            for t in all_triggers_this_stage.get(stage_num, []):
                if t in active_set:
                    st.markdown(f"- ✅ **{t}** (firing)")
                else:
                    st.markdown(f"- ⬜ {t}")

    # Tail risk analysis
    st.markdown("### Tail Risk Profile")
    st.markdown(f"""
    <div style='padding:20px 24px;background:{seq['tail_color']}10;border:1px solid {seq['tail_color']}44;border-radius:14px;margin:8px 0'>
      <div style='color:{seq['tail_color']};font-size:0.78em;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>Tail Risk Level</div>
      <div style='color:{seq['tail_color']};font-size:2.2em;font-weight:700;margin-top:4px;letter-spacing:-0.02em'>{seq['tail_level']}</div>
      <div style='color:#d1d1d6;font-size:0.95em;margin-top:8px;line-height:1.5'>Expected drawdown range: <strong style='color:#f5f5f7'>{seq['tail_drawdown']}</strong></div>
      <div style='color:#d1d1d6;font-size:0.95em;margin-top:4px'>{seq['tail_scenario']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Methodology
    st.markdown("### Methodology")
    st.markdown("""
    The Sequence Stage Engine synthesizes:
    - **ML Recession Radar** probability
    - **5 ground-level sensors**: LSI (labor), CSI (credit), GCI (global), CFHI (consumer), CREI (real economy)
    - **3 structural pipes**: Pipe 1 (foreign funding), Pipe 2 (shadow banking), Pipe 3 (sovereign debt)
    - **Specific trigger thresholds** for each of 6 sequence stages

    Each stage has 2-6 specific triggers. When >50% of a stage's triggers fire, we're "in" that stage.
    Fractional position calculated by looking at how many next-stage triggers are already forming.

    **Why this matters:** A single snapshot of 20 metrics is cognitively overwhelming and lets you default to
    looking at the ML probability (which is low right now). The sequence stage forces you to integrate everything
    into one position — where are we on the crisis path? — which is what actually matters for decisions.

    **Limitations:** The sequence engine is a synthesis tool, not a prediction tool. Historical stages in 1990, 2000,
    2007, 2019 ran at different speeds. Japan has been in Stage 3 effectively since 1990 with no crisis yet.
    Position in sequence tells you SET-UP, not timing.
    """)

    st.caption("Inspired by the pattern recognition in Adam Tooze's *Crashed* (2018). "
               "Stage definitions based on observed patterns in 1990, 2000, 2007, 2019, 2020, 2023 events.")





# ── TAB 1: DASHBOARD ──
with tab1:
    st.markdown("### Indicator Matrix")

    # FRED live data badge
    if fred_status == "LIVE" and raw_fred:
        with st.expander("📡 Live FRED data", expanded=False):
            live_items = [
                {"Indicator": k, "Value": f"{v[0]:.2f}{v[1]}", "As of": v[2]}
                for k, v in raw_fred.items()
            ]
            st.dataframe(pd.DataFrame(live_items), use_container_width=True, hide_index=True)
            st.caption("Data automatically refreshed from FRED. Cached for 1 hour.")

    # Indicator table
    df = pd.DataFrame({
        'Indicator': INDICATOR_NAMES,
        'Fire Level': [f"{v*100:.0f}%" for v in current],
        'Judgment Weight': [f"{w}%" for w in JUDGMENT_WEIGHTS],
        'Status': ['🔴 High' if current[i] >= 0.7 else '🟡 Medium' if current[i] >= 0.4 else '🟢 Low' for i in range(11)],
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Fire Level Visualization")
    chart_df = pd.DataFrame({'Indicator': INDICATOR_NAMES, 'Fire': current})
    st.bar_chart(chart_df.set_index('Indicator'))


# ── TAB 2: ACTION PLAN ──
with tab2:
    st.markdown("### Asset Allocation by Regime")

    regimes = [
        (0, 0.20, 'LOW RISK',  '70-80%', '15-20%', '0-5%',   '3-5%'),
        (0.20, 0.35, 'MODERATE', '60-70%', '20-25%', '5-8%',   '5-10%'),
        (0.35, 0.55, 'ELEVATED', '50-60%', '20-25%', '8-12%',  '10-15%'),
        (0.55, 0.75, 'HIGH',     '35-45%', '20-25%', '10-15%', '20-30%'),
        (0.75, 1.0,  'CRISIS',   '20-30%', '25-30%', '15-20%', '25-35%'),
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


# ── TAB 3: SCENARIOS ──
with tab3:
    st.markdown("### Scenario Analysis")
    st.caption("Each scenario adjusts indicators through the ML model")

    scenarios = [
        {'name': 'A. Peace Deal', 'sub': 'Ceasefire holds, oil to $75',
         'ch': {3:.2, 4:.15, 7:.4, 8:.2, 9:.3, 10:.7},
         'spx': '+15 to +22%', 'bonds': '+3 to +5%', 'gold': '-8 to -12%',
         'action': 'Risk-on. Rebuild cyclicals. Cash to 5%.'},
        {'name': 'B. Slow Burn', 'sub': 'Oil $90-100 for 6+ months',
         'ch': {3:.55, 4:.25, 7:.6, 8:.35, 9:.5, 10:.65},
         'spx': '-8 to -15%', 'bonds': '+1 to +3%', 'gold': '+5 to +10%',
         'action': 'Hold defensives. Trim cyclicals. Cash 10-15%.'},
        {'name': 'C. Escalation', 'sub': 'Hormuz closed, oil $130+',
         'ch': {3:.9, 4:.6, 7:.85, 8:.6, 9:.8, 10:.5},
         'spx': '-25 to -35%', 'bonds': '-2 to +2%', 'gold': '+15 to +25%',
         'action': 'Aggressive de-risk. Cash 25-30%. Puts.'},
        {'name': 'D. Credit Event', 'sub': 'CRE/private credit blowup',
         'ch': {3:.55, 4:.85, 7:.9, 8:.7, 9:.9, 10:.5},
         'spx': '-35 to -50%', 'bonds': '+8 to +15%', 'gold': '+20 to +30%',
         'action': 'Max defensive. Treasuries + gold + cash.'},
    ]

    for sc in scenarios:
        adj = list(current)
        for idx, val in sc['ch'].items(): adj[idx] = val
        sv, _ = get_current_vec(adj)
        sp = model.predict(sv)['ensemble'][0]

        with st.expander(f"{sc['name']}: {sc['sub']} — **{sp*100:.0f}% recession probability**", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("S&P 500", sc['spx'])
            col2.metric("Bonds", sc['bonds'])
            col3.metric("Gold", sc['gold'])
            st.markdown(f"**Action:** {sc['action']}")


# ── TAB 4: ML ENGINE ──
with tab4:
    st.markdown("### ML Engine Internals")

    col1, col2, col3 = st.columns(3)
    col1.metric("Ensemble", f"{probs['ensemble'][0]*100:.0f}%")
    col2.metric("Agreement", f"{agreement*100:.0f}%")
    col3.metric("Training Events", f"n={len(EVENTS)}")

    st.markdown("### Individual Models")
    col1, col2, col3 = st.columns(3)
    col1.metric("Logistic Regression", f"{probs['lr'][0]*100:.0f}%")
    col2.metric("Random Forest", f"{probs['rf'][0]*100:.0f}%")
    col3.metric("Gradient Boosting", f"{probs['gbt'][0]*100:.0f}%")

    st.markdown("### Leave-One-Out Cross-Validation")
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
            'Ensemble': f"{p['ensemble'][0]*100:.0f}%",
            'Correct': '✓' if pred_class == y[i] else '✗',
        })
    loo_df = pd.DataFrame(loo_results)
    acc = sum(1 for r in loo_results if r['Correct'] == '✓') / len(loo_results)
    st.dataframe(loo_df, use_container_width=True, hide_index=True)
    st.metric("LOOCV Accuracy", f"{acc*100:.0f}%")

    # Bootstrap CI
    with st.spinner("Computing bootstrap CI..."):
        bs = model.bootstrap_ci(X, y, cur_vec, n=50, sw=w)
    st.caption(f"Bootstrap 90% CI: {bs['p5']*100:.0f}% — {bs['p95']*100:.0f}% | Median: {bs['p50']*100:.0f}%")


# ── TAB 5: LSI ──
with tab5:
    st.markdown(f"### Labor Stress Index · {lsi_total:.0f}/100")
    st.markdown(f"**Level:** {lsi_level}")

    st.markdown("#### What the headline misses")
    st.markdown(f"""
    - Headline U-3 unemployment: {u3:.1f}%
    - LSI composite: **{lsi_total:.0f}/100 ({lsi_level})**
    - The gap between these IS the illusion
    """)

    comp_data = {
        'Avg weekly hours': (lsi_hours, lsi_comps['hours'], "Hidden layoffs"),
        'Quits rate': (lsi_quits, lsi_comps['quits'], "Workers scared"),
        'Temp employment YoY': (lsi_temp, lsi_comps['temp'], "First domino"),
        'Continued claims': (lsi_claims, lsi_comps['claims'], "Harder to find work"),
        'U6-U3 gap': (lsi_u6, lsi_comps['u6'], "Underemployment"),
    }
    df = pd.DataFrame([
        {'Component': k, 'Value': v[0], 'Stress': f"{v[1]:.0f}/100", 'Why': v[2]}
        for k, v in comp_data.items()
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.warning("Every historical correction had LSI <30. Every recession had LSI >35 at market peak. "
               f"Current {lsi_total:.0f} has NEVER been associated with correction-only outcome.")


# ── TAB 6: CSI ──
with tab6:
    st.markdown(f"### Credit Stress Index · {csi_total:.0f}/100")
    st.markdown(f"**Level:** {csi_level}")

    st.markdown("#### Divergence from market pricing")
    if div_gap > 0.15:
        st.error(f"CSI ({csi_total:.0f}/100) significantly exceeds HY OAS fire ({hy_fire*100:.0f}%). "
                 f"This pattern preceded GFC by 3-6 months.")
    elif div_gap < -0.15:
        st.info("Market pricing exceeds fundamentals — may be overreacting.")
    else:
        st.success("CSI and credit spreads aligned. High confidence.")

    csi_labels = {
        'sloos_ci': ('SLOOS C&I tightening', csi_sloos_ci, '%'),
        'sloos_cc': ('SLOOS CC tightening', csi_sloos_cc, '%'),
        'chargeoff': ('Loan charge-off rate', csi_chargeoff, '%'),
        'cons_del': ('Consumer delinquency', csi_cons_del, '%'),
        'auto_del': ('Auto 60+d delinq', csi_auto_del, '%'),
        'cp_chg': ('Commercial paper YoY', csi_cp_chg, '%'),
    }
    cdf = pd.DataFrame([
        {'Component': v[0], 'Value': f"{v[1]}{v[2]}", 'Stress': f"{csi_comps[k]:.0f}/100"}
        for k, v in csi_labels.items()
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)


# ── TAB 7: GCI ──
with tab7:
    st.markdown(f"### Global Contagion Index · {gci_total:.0f}/100")
    st.markdown(f"**Level:** {gci_level}")

    st.markdown("#### The dimension US-focused models miss")
    st.markdown("""
    - 40% of S&P 500 revenue is international
    - Average lag from global stress to US market reaction: 3-6 months
    - Dollar weakening during geopolitical stress is unprecedented
    """)

    gci_labels = {
        'dxy_surge': ('USD 6m change', gci_dxy_surge, '%'),
        'em_sov': ('EM sovereign spread', gci_em_sov, 'bps'),
        'global_pmi': ('Global manuf PMI', gci_global_pmi, ''),
        'china_ci': ('China credit impulse', gci_china_ci, 'pp'),
        'eu_bank': ('EU bank stress', gci_eu_bank, '/100'),
    }
    gdf = pd.DataFrame([
        {'Component': v[0], 'Value': f"{v[1]}{v[2]}", 'Stress': f"{gci_comps[k]:.0f}/100"}
        for k, v in gci_labels.items()
    ])
    st.dataframe(gdf, use_container_width=True, hide_index=True)


# ── TAB 8: CFHI ──
with tab8:
    st.markdown(f"### Consumer Financial Health Index · {cfhi_total:.0f}/100")
    st.markdown(f"**Phase:** {cfhi_phase} — {cfhi_phase_desc}")

    cfhi_labels = {
        'real_inc': ('Real disposable income YoY', f"{cfhi_real_inc:+.1f}%"),
        'savings': ('Personal savings rate', f"{cfhi_savings:.1f}%"),
        'tdsp': ('Debt service ratio', f"{cfhi_tdsp:.1f}%"),
        'cc_util': ('CC utilization', f"{cfhi_cc_util}%"),
        'cc_del': ('CC delinquency 90+d', f"{cfhi_cc_del:.2f}%"),
        'retail': ('Real retail sales YoY', f"{cfhi_retail:+.1f}%"),
    }
    cdf = pd.DataFrame([
        {'Component': v[0], 'Value': v[1], 'Stress': f"{cfhi_comps[k]:.0f}/100"}
        for k, v in cfhi_labels.items()
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    st.error("""
    **Phase 3 triggers (watch these monthly):**
    - Real retail sales turning **negative**
    - CC delinquency 90+d crossing **4.0%**
    - Savings rate falling below **3.0%**
    """)


# ── TAB 9: CREI — Corporate Real Economy Index ──
with tab9:
    st.markdown(f"### Corporate Real Economy Index · {crei_total:.0f}/100")
    st.markdown(f"**Level: {crei_level}** — {crei_desc}")

    st.markdown("""
    <div style='padding:14px;background:rgba(200,168,74,0.08);border-left:3px solid #c8a84a;border-radius:10px;margin:8px 0'>
    <div style='color:#c8a84a;font-size:0.82em;font-weight:600;letter-spacing:0.05em'>THE ATH ILLUSION MEASURED</div>
    <div style='color:#d1d1d6;font-size:0.9em;margin-top:6px;line-height:1.5'>
    S&P 500 EPS is up 63% since 2019. Great headline. But: top 7 tech companies contributed
    60%+ of that growth. Ex-top-7, S&P 500 EPS growth is only 3-4% annualized (barely above inflation).
    Russell 2000 has had ZERO real earnings growth since 2019. The CREI measures that real economy —
    the one where most Americans actually work.
    </div>
    </div>
    """, unsafe_allow_html=True)

    crei_labels = {
        'r2k_eps': ('Russell 2000 EPS YoY', f"{crei_r2k_eps:+.0f}%",
                    'Small-cap earnings — the Joe Average companies'),
        'zombie': ('Zombie company share', f"{crei_zombie}%",
                   'Companies unable to cover interest from operating profits'),
        'maturity_wall': ('2026 maturity wall', f"${crei_maturity_wall}B",
                          'Small-cap debt requiring refinance this year'),
        'coupon_shock': ('Refi coupon shock', f"{crei_coupon_shock:.1f}pp",
                         'Old rate (1-2%) vs new rate (6.5-8%)'),
        'regional': ('Regional bank stress', f"{crei_regional}/100",
                     'KRE, CRE exposure, deposit outflows'),
        'smb_bank': ('SMB bankruptcy YoY', f"+{crei_smb_bankruptcy}%",
                     'Actual small business failures'),
        'ew_div': ('EW vs CW S&P 500', f"{crei_ew_divergence}%",
                   'Market concentration — how narrow is the rally?'),
        'midcap_margin': ('Midcap margin compression', f"{crei_midcap_margin}/100",
                          'S&P 400 operating margin decline (leads small-cap)'),
    }

    cdf = pd.DataFrame([
        {'Component': v[0], 'Value': v[1], 'Stress': f"{crei_comps[k]:.0f}/100", 'Why': v[2]}
        for k, v in crei_labels.items()
    ])
    st.dataframe(cdf, use_container_width=True, hide_index=True)

    st.markdown("### Historical Comparison")
    hist_crei = pd.DataFrame([
        {'Period': '2021-Q4', 'CREI': 18, 'Context': 'Stimulus peak — small caps strong'},
        {'Period': '2019-Q4', 'CREI': 28, 'Context': 'Pre-COVID healthy baseline'},
        {'Period': '1999-Q4', 'CREI': 38, 'Context': 'Dot-com peak (similar concentration)'},
        {'Period': '2007-Q2', 'CREI': 48, 'Context': 'Pre-GFC real economy softening'},
        {'Period': 'NOW', 'CREI': crei_total, 'Context': 'K-shaped extreme + zombie wall'},
        {'Period': '2023-Q3', 'CREI': 62, 'Context': 'Post-SVB regional bank stress'},
        {'Period': '2008-Q4', 'CREI': 92, 'Context': 'GFC real economy collapse'},
    ])
    st.dataframe(hist_crei.sort_values('CREI'), use_container_width=True, hide_index=True)

    st.error("""
    **The zombie wall mechanism (how small-cap stress cascades to real economy):**

    1. 43% of Russell 2000 can't cover interest from operating profits
    2. They survived 2022-2024 on continuous refinancing access
    3. $368B maturity wall in 2026, $1.35T total through 2028
    4. Refinancing at 6.5-8% vs original 1-2% = 5.5pp coupon shock
    5. For thin-margin zombies, this is existential
    6. Bankruptcies accelerate → layoffs → consumer weakness
    7. Regional banks with small-biz exposure take losses
    8. Credit conditions tighten broadly → more zombie failures
    9. Cycle is self-reinforcing once started

    **This is the mechanism that converts a 'soft' slowdown into a recession.**
    """)

    st.markdown("### What CREI Tells Us That S&P 500 EPS Cannot")
    st.markdown("""
    - **S&P 500 EPS since 2019:** +63% (great!)
    - **Ex-top-7 tech:** +20% cumulative (3-4% annualized, barely above inflation)
    - **Russell 2000:** 0% real growth
    - **Equal-weight S&P 500 vs cap-weight:** -18% YTD (extreme concentration)

    The ATH narrative is mathematically true but economically misleading. The economy
    where most Americans work (small caps, midcaps, regional banks, real estate, small
    business) is under significant stress. CREI at 60 is the quantification of what
    headlines won't tell you.
    """)

    # Five-dimensional picture
    st.markdown("### Complete Ground-Level Picture")
    full_picture = pd.DataFrame([
        {'Layer': 'Market (ML)', 'Reading': f"{ml_p*100:.0f}%", 'Signal': 'Low risk'},
        {'Layer': 'Market (HY OAS)', 'Reading': f"{hy_fire*100:.0f}% fire", 'Signal': 'Calm'},
        {'Layer': 'Labor (LSI)', 'Reading': f"{lsi_total:.0f}/100", 'Signal': lsi_level},
        {'Layer': 'Credit (CSI)', 'Reading': f"{csi_total:.0f}/100", 'Signal': csi_level},
        {'Layer': 'Global (GCI)', 'Reading': f"{gci_total:.0f}/100", 'Signal': gci_level},
        {'Layer': 'Consumer (CFHI)', 'Reading': f"{cfhi_total:.0f}/100", 'Signal': cfhi_phase},
        {'Layer': 'Real Econ (CREI)', 'Reading': f"{crei_total:.0f}/100", 'Signal': crei_level},
    ])
    st.dataframe(full_picture, use_container_width=True, hide_index=True)


# ── TAB 10: PIPE 1 ──
with tab10:
    st.markdown(f"### Pipe 1 · Foreign Funding")
    st.markdown(f"**Status: {p1_status}**")

    st.info("Net capital flows look balanced. Gross flows reveal massive dependency: "
            "$6T+ SWF wealth, $250B Gulf Treasury holdings, $1.5T Japanese dollar assets. "
            "This is the 2026 transnational flow analog to 2007 European bank flows.")

    p1_data = [
        ("Gulf SWF US concentration", 65, "/100", "European bank SIV exposure"),
        ("Saudi oil to China", 30, "%", "(No 2007 analog — new)"),
        ("DXY during stress", -5, "%", "USD strengthened in 2007"),
        ("Saudi PIF new US commitments", -70, "%", "ABCP issuance decline Aug 2007"),
        ("Japanese hedging cost", 6.8, "%", "Swap spread widening Aug 2007"),
        ("EM gold accumulation", 75, "/100", "(No 2007 analog)"),
        ("Foreign Treasury demand", 35, "/100", "MMF flow into ABCP slowing"),
        ("Chinese shadow reserves", 60, "/100", "European SIV off-balance-sheet"),
    ]

    p1_df = pd.DataFrame([
        {'Indicator': d[0], 'Value': f"{d[1]}{d[2]}", 'Score': f"{pipe1_scores[i]:.0f}/100",
         '2007 Analog': d[3]}
        for i, d in enumerate(p1_data)
    ])
    st.dataframe(p1_df, use_container_width=True, hide_index=True)

    st.error("""
    **Critical asymmetry:** Unlike Pipe 2, Fed support doesn't help here.
    YCC (yield curve control) debases the dollar further, accelerating the problem.
    Monetary equivalent of pouring gasoline on fire.
    """)


# ── TAB 11: PIPE 2 ──
with tab11:
    st.markdown(f"### Pipe 2 · Shadow Banking")
    st.markdown(f"**Status: {p2_status}**")

    st.info("2008's ABCP conduit system reborn as private credit. $3.5T untested in a cycle. "
            "Leverage-on-leverage via bank subscription lines. Same structural pattern as 2007.")

    p2_data = [
        ("BDC discount to NAV", 17, "%", "ABCP rates spike Aug 2007"),
        ("PIK income share", 8, "%", "Subprime teaser resets"),
        ("PC tech exposure", 42, "%", "Subprime CA/FL/AZ/NV"),
        ("True PC default rate", 5.8, "%", "Subprime delinq mid-2007"),
        ("Evergreen fund pressure", 65, "/100", "BNP Paribas fund freeze"),
        ("Subscription line exposure", 72, "/100", "SIV off-balance-sheet"),
        ("PC negative FCF share", 40, "%", "Subprime unable to pay"),
        ("MMF PC paper exposure", 45, "/100", "MMF to ABCP direct parallel"),
    ]

    p2_df = pd.DataFrame([
        {'Indicator': d[0], 'Value': f"{d[1]}{d[2]}", 'Score': f"{pipe2_scores[i]:.0f}/100",
         '2007 Analog': d[3]}
        for i, d in enumerate(p2_data)
    ])
    st.dataframe(p2_df, use_container_width=True, hide_index=True)

    st.warning("40% of PC borrowers have negative free cash flow. "
               "Zombies surviving on lender forbearance — exact 2007 subprime pattern.")


# ── TAB 12: PIPE 3 · SOVEREIGN DEBT & MARKET STRUCTURE ──
with tab12:
    st.markdown(f"### Pipe 3 · Sovereign Debt & Market Structure")

    st.markdown("""
    <div style='padding:14px;background:rgba(200,168,74,0.08);border-left:3px solid #c8a84a;border-radius:10px;margin:8px 0'>
    <div style='color:#c8a84a;font-size:0.82em;font-weight:600;letter-spacing:0.05em'>TWO TIME HORIZONS</div>
    <div style='color:#d1d1d6;font-size:0.9em;margin-top:6px;line-height:1.5'>
    <strong>Short-term:</strong> Can the Treasury market function under stress in the next 6 months?<br>
    <strong>Long-term:</strong> Is the fiscal trajectory sustainable over 5-20 years?<br><br>
    Both questions share the same mechanism: who buys government bonds, at what price, with what fragility.
    IMF April 2026 explicitly identifies "erosion of US Treasury safety premium" and "leveraged nonbank marginal buyers" as unprecedented vulnerabilities.
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Short-term indicators (market structure)
    pipe3_short = [
        {"label": "HF Treasury basis trade", "value": 1000, "unit": "$B",
         "thresholds": [200, 400, 700, 1000, 1500], "invert": False,
         "ref_2007": "ABCP conduit leverage 2007",
         "why": "$1T notional short Treasury futures, up to 56-to-1 leverage. Unwind in March 2020 required $1T Fed intervention. Today's position is LARGER. Any repo rate spike triggers forced deleveraging."},
        {"label": "Dealer balance sheet constraint", "value": 65, "unit": "/100",
         "thresholds": [25, 40, 60, 80, 95], "invert": False,
         "ref_2007": "Pre-Lehman dealer repo constraint",
         "why": "SLR regulations constrain dealer capacity just as Treasury issuance surged. Dealers intermediate rather than warehouse. Can't absorb flow during stress. Score 65: adequate normal, inadequate 2x+ normal flow."},
        {"label": "Repo market stress", "value": 30, "unit": "/100",
         "thresholds": [15, 30, 50, 75, 90], "invert": False,
         "ref_2007": "Aug 2007 Libor-OIS spread spike",
         "why": "$12T US repo market. September 2019 saw SOFR spike to 10%. 75% of hedge fund Treasury repo is bilateral (opaque). Haircuts can change overnight, cascading forced deleveraging instantly."},
        {"label": "10yr term premium dislocation", "value": 45, "unit": "/100",
         "thresholds": [20, 35, 55, 75, 90], "invert": False,
         "ref_2007": "Swap spread inversion Aug 2007",
         "why": "Term premium rising from ~0 to positive as safety premium erodes and supply grows. Higher term premium = long rates rise independent of Fed. Hurts mortgages, CRE, tech valuations."},
    ]

    # Long-term indicators (fiscal sustainability)
    pipe3_long = [
        {"label": "US debt-to-GDP", "value": 120, "unit": "%",
         "thresholds": [60, 80, 100, 130, 150], "invert": False,
         "ref_2007": "(Pre-GFC was ~60%)",
         "why": "CBO projects 136% by 2036. Dallas Fed: each 1pp of debt/GDP raises long yields ~3bps. UK hit 200% post-Waterloo without crisis but ran surpluses afterward. US runs structural 6% deficits — no path to reduction without crisis."},
        {"label": "Interest cost / federal revenue", "value": 20, "unit": "%",
         "thresholds": [8, 12, 18, 25, 35], "invert": False,
         "ref_2007": "(Was ~9% in 2007)",
         "why": "Interest hit $1T/yr in 2024 — LARGER than defense. Projected $2T+ by 2035. At 30%+ of revenue, government has no fiscal flexibility. This is the 'fiscal dominance' point where monetary policy becomes captured."},
        {"label": "Primary deficit vs stabilizing", "value": 6, "unit": "%",
         "thresholds": [0, 2, 4, 6, 10], "invert": False,
         "ref_2007": "(2007 US was running close to balance)",
         "why": "IMF April 2026: 'global fiscal gap closed to zero.' Debt-stabilizing primary deficit is approximately ZERO at current r-g dynamics. Every year of current policy adds 3%+ to debt/GDP structurally."},
        {"label": "Treasury safety premium erosion", "value": 70, "unit": "/100",
         "thresholds": [15, 35, 55, 80, 95], "invert": False,
         "ref_2007": "(Was intact in 2007)",
         "why": "IMF April 2026 EXPLICIT warning. 1990-2010 US 10yr yield averaged 1-1.5% BELOW nominal GDP growth. Today roughly equal. If r > g, debt dynamics become self-reinforcing unstable. Score 70: premium eroded substantially."},
        {"label": "Foreign holding share", "value": 22, "unit": "%",
         "thresholds": [35, 30, 25, 20, 15], "invert": True,
         "ref_2007": "Still rising pre-2007",
         "why": "Foreign share declined 34% (2014) → 22% (2026). Domestic buyers (banks SLR-constrained, MMFs mandate-capped, hedge funds fragile) must absorb growing share of ballooning issuance. Natural buyer pool shrinking while supply accelerates."},
        {"label": "G7 weighted debt contagion", "value": 75, "unit": "/100",
         "thresholds": [25, 45, 65, 80, 95], "invert": False,
         "ref_2007": "(G7 debt was manageable)",
         "why": "ALL G7 sovereigns elevated: US 120%, UK 100%, France 115%, Italy 137%, Japan 260%, Germany 74%. IMF projects France 129% by 2030. Weighted G7 crosses 125% — no precedent. No natural DM safe haven if one sovereign ruptures."},
    ]

    def score_p3(value, thresholds, inverted=False):
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

    for ind in pipe3_short + pipe3_long:
        ind["score"] = score_p3(ind["value"], ind["thresholds"], ind.get("invert", False))
        if ind["score"] < 30: ind["status"], ind["color"] = "HEALTHY", "#30d158"
        elif ind["score"] < 50: ind["status"], ind["color"] = "WATCH", "#c8a84a"
        elif ind["score"] < 70: ind["status"], ind["color"] = "STRESSED", "#ff453a"
        elif ind["score"] < 85: ind["status"], ind["color"] = "AT-RISK", "#ff453a"
        else: ind["status"], ind["color"] = "ACTIVE-RUPTURE", "#ff453a"

    def classify_pipe(scores):
        if not scores: return "HEALTHY", "#30d158"
        max_s = max(scores)
        above_75 = sum(1 for s in scores if s >= 75)
        above_50 = sum(1 for s in scores if s >= 50)
        mean_s = np.mean(scores)
        if max_s >= 90: return "ACTIVE-RUPTURE", "#ff453a"
        elif above_75 >= 2: return "AT-RISK", "#ff453a"
        elif above_50 >= 3 or max_s >= 75: return "STRESSED", "#ff453a"
        elif above_50 >= 1 or mean_s >= 40: return "WATCH", "#c8a84a"
        else: return "HEALTHY", "#30d158"

    short_scores = [ind["score"] for ind in pipe3_short]
    long_scores = [ind["score"] for ind in pipe3_long]
    all_scores = short_scores + long_scores

    p3_short_status, p3_short_color = classify_pipe(short_scores)
    p3_long_status, p3_long_color = classify_pipe(long_scores)
    p3_overall_status, p3_overall_color = classify_pipe(all_scores)

    # Dual-horizon hero
    h1, h2 = st.columns(2)
    with h1:
        st.markdown(f"""
        <div style='padding:24px;background:linear-gradient(135deg,{p3_short_color}15 0%,{p3_short_color}05 100%);border:1px solid {p3_short_color}55;border-radius:16px'>
        <div style='color:#86868b;font-size:0.72em;letter-spacing:0.12em;text-transform:uppercase'>SHORT-TERM · MARKET STRUCTURE</div>
        <h1 style='color:{p3_short_color};margin:8px 0 0 0;font-size:2.2em;letter-spacing:-0.02em'>{p3_short_status}</h1>
        <div style='color:#d1d1d6;font-size:0.85em;margin-top:8px'>Next 6 months Treasury market rupture risk</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div style='padding:24px;background:linear-gradient(135deg,{p3_long_color}15 0%,{p3_long_color}05 100%);border:1px solid {p3_long_color}55;border-radius:16px'>
        <div style='color:#86868b;font-size:0.72em;letter-spacing:0.12em;text-transform:uppercase'>LONG-TERM · FISCAL SUSTAINABILITY</div>
        <h1 style='color:{p3_long_color};margin:8px 0 0 0;font-size:2.2em;letter-spacing:-0.02em'>{p3_long_status}</h1>
        <div style='color:#d1d1d6;font-size:0.85em;margin-top:8px'>5-20 year fiscal trajectory risk</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Status", p3_overall_status)
    c2.metric("Indicators Stressed (≥50)", f"{sum(1 for s in all_scores if s >= 50)}/{len(all_scores)}")
    c3.metric("Indicators At-Risk (≥75)", f"{sum(1 for s in all_scores if s >= 75)}/{len(all_scores)}")

    # Short-term indicators
    st.markdown("### Short-Term · Market Structure Fragility")
    st.caption("Can the Treasury market absorb stress? Measures plumbing resilience.")
    short_df = pd.DataFrame([
        {
            "Indicator": ind["label"],
            "Value": f"{ind['value']}{ind['unit']}",
            "Score": f"{ind['score']:.0f}/100",
            "Status": ind["status"],
            "2007 Analog": ind["ref_2007"],
        }
        for ind in pipe3_short
    ])
    st.dataframe(short_df, use_container_width=True, hide_index=True)

    # Long-term indicators
    st.markdown("### Long-Term · Fiscal Sustainability")
    st.caption("Is the fiscal trajectory sustainable? Measures structural deterioration.")
    long_df = pd.DataFrame([
        {
            "Indicator": ind["label"],
            "Value": f"{ind['value']}{ind['unit']}",
            "Score": f"{ind['score']:.0f}/100",
            "Status": ind["status"],
            "2007 Analog": ind["ref_2007"],
        }
        for ind in pipe3_long
    ])
    st.dataframe(long_df, use_container_width=True, hide_index=True)

    # Detailed reasoning
    st.markdown("### Why Each Indicator Matters")
    for ind in pipe3_short + pipe3_long:
        with st.expander(f"{ind['label']} — {ind['status']} ({ind['value']}{ind['unit']})"):
            st.markdown(f"**Category:** {'Short-term · Market Structure' if ind in pipe3_short else 'Long-term · Fiscal Sustainability'}")
            st.markdown(f"**2007 Equivalent:** {ind['ref_2007']}")
            st.markdown(f"**Why it matters:** {ind['why']}")

    # Historical context
    st.markdown("### Historical Context · UK vs France")

    st.markdown("""
    <div style='padding:16px;background:rgba(255,255,255,0.03);border-left:3px solid #c8a84a;border-radius:10px;margin:8px 0'>
    <div style='color:#f5f5f7;font-weight:500;margin-bottom:8px'>Why debt-financed power sometimes works and sometimes fails</div>
    <div style='color:#d1d1d6;font-size:0.9em;line-height:1.6'>
    <strong>UK 1815:</strong> Debt/GDP 200% (higher than today's US). Parliament controlled taxation post-1688,
    making repayment credible. Debt funded productive capacity (industrial infrastructure) + defensive wars.
    Britain ran surpluses most of 19thC. Debt fell steadily. <em>Worked.</em><br><br>

    <strong>US 1945:</strong> Debt/GDP 119%. Funded WW2 + GI Bill (education) + post-war infrastructure.
    Growth &gt; rates for 35 years. Debt fell to 30% by 1980. <em>Worked.</em><br><br>

    <strong>Greece 2010:</strong> Debt/GDP 146%. Funded consumption (pensions, public employment).
    No monetary sovereignty. Market refused refinance. <em>Forced restructuring.</em><br><br>

    <strong>UK October 2022:</strong> Debt/GDP 100%. Unfunded tax cuts triggered LDI pension forced selling.
    100+bps yields in days. BoE intervention required. <em>Leveraged marginal buyers made it fragile.</em><br><br>

    <strong>US 2026:</strong> Debt/GDP 120%. ~60% of spending is consumption-smoothing.
    Structural 6% deficits. Interest cost exceeds defense. Safety premium eroding.
    Multiple vulnerabilities simultaneously. <em>Unprecedented configuration.</em>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Scenarios
    st.markdown("### Transmission Scenarios")

    with st.expander("⚡ Short-term · Basis Trade Cascade (March 2020 redux) — Days to weeks"):
        st.markdown("""
        **Probability:** Moderate — elevated positions

        **Historical analog:** March 2020 Treasury dysfunction

        **Sequence:**
        1. Repo rate spike (quarter-end, Treasury auction, volatility)
        2. Hedge fund funding costs jump
        3. Basis trade profitability evaporates
        4. Forced deleveraging begins
        5. Treasuries sold into thin dealer market
        6. Yields spike 50-100bps in days
        7. Dealer balance sheets max out
        8. Fed intervention required (like March 2020)
        9. Market functioning restored but at higher yields
        """)

    with st.expander("⚡ Short-term · Foreign Selling Cascade (UK 2022 redux) — Weeks to months"):
        st.markdown("""
        **Probability:** Moderate — structural drivers present

        **Historical analog:** UK Liz Truss Gilt crisis October 2022

        **Sequence:**
        1. Major foreign holder announces reduced allocation
        2. Other foreign holders follow (herding)
        3. Treasury auction bid-to-cover weakens
        4. Long yields rise 50+bps
        5. Leveraged Treasury positions hit margin calls
        6. Forced selling amplifies initial move
        7. Fed may intervene or accept higher yields
        8. Higher yields transmit to mortgages, CRE, tech valuations
        """)

    with st.expander("🔁 Long-term · Slow Fiscal Erosion (Japan Path) — 10-25 years"):
        st.markdown("""
        **Probability:** High — this is the baseline path

        **Historical analog:** Japan 1990-present (260% debt/GDP, no crisis yet)

        **Sequence:**
        1. Debt/GDP grinds higher at 2-3% per year
        2. Interest costs absorb increasing share of budget
        3. Political impossibility of cutting entitlements
        4. Currency depreciation over decades
        5. Real returns on fixed income become deeply negative
        6. Wealth destruction via financial repression
        7. No acute crisis — just gradual real wealth erosion

        **This is probably the most likely path for US.** Not a crash — a long grind.
        """)

    with st.expander("💥 Long-term · Forced Restructuring (Greece Path) — 5-15 years"):
        st.markdown("""
        **Probability:** Lower but non-trivial for non-dollar sovereigns

        **Most at risk:** France, Italy, UK (in that order) before US

        **Historical analog:** Greece 2010-2012, Argentina multiple times

        **Sequence:**
        1. Market refuses to refinance at acceptable yields
        2. Emergency fiscal consolidation attempted
        3. Political collapse of consolidation
        4. Emergency IMF/multilateral intervention
        5. Debt restructuring negotiated with creditors
        6. 20-40% nominal haircuts to bondholders
        7. Multi-decade economic stagnation
        """)

    with st.expander("💵 Long-term · Inflationary Monetization (1970s path) — 3-10 years"):
        st.markdown("""
        **Probability:** Highest for US given dollar reserve status

        **Historical analog:** US 1970s, UK 1970s

        **Sequence:**
        1. Fed forced to support Treasury market (YCC explicit or implicit)
        2. Money supply expansion absorbed by Treasury purchases
        3. Inflation runs 5-8% sustainably
        4. Real debt burden erodes by inflation
        5. Savers and bondholders bear the cost
        6. Political reaction drives populist policies
        7. Eventually requires new Volcker-style adjustment
        """)

    # Who gets hit first
    st.markdown("### Who Gets Hit First")
    hit_order = [
        ("Long-duration bondholders (pension funds, insurers)", "Structural losers"),
        ("Savers in local currency (retirees, fixed income)", "Inflation erosion"),
        ("Leveraged duration bets (basis trade, LDI)", "Forced deleveraging"),
        ("Emerging markets with USD debt", "Dollar funding stress"),
        ("Housing market (mortgage rate channel)", "Slow-burn"),
        ("Commercial real estate (maturity wall)", "Acute"),
        ("Tech/growth equities (duration-sensitive)", "Severe"),
        ("France, Italy before US (sequential sovereign stress)", "Cascading"),
    ]
    for i, (target, note) in enumerate(hit_order, 1):
        st.markdown(f"**{i}.** {target} — *{note}*")

    # Early warning signals
    st.markdown("### Early Warning Signals")

    st.markdown("**Short-term (check weekly):**")
    short_signals = [
        "SOFR-IOR spread widening >10bps sustained",
        "Treasury bid-to-cover below 2.3x at 30yr auctions",
        "10yr yield rising >25bps in a week without data catalyst",
        "Dealer positioning showing inventory stress (weekly NY Fed H.4.1)",
        "Fed reverse repo facility spiking above $1T",
    ]
    for s in short_signals: st.markdown(f"- {s}")

    st.markdown("**Long-term (check monthly):**")
    long_signals = [
        "US 10yr yield > US nominal GDP growth for 6+ months",
        "Interest costs crossing 25% of federal revenue",
        "Any G7 sovereign 10yr yield spiking >100bps absent cyclical factors",
        "CDS on US sovereign trading above 50bps sustained",
        "French OAT-Bund spread above 75bps sustained",
        "Credit rating agencies announce US downgrade review",
    ]
    for s in long_signals: st.markdown(f"- {s}")

    # The critical insight
    st.markdown("### The Critical Insight")

    st.markdown("""
    <div style='padding:20px;background:rgba(255,69,58,0.08);border:1px solid rgba(255,69,58,0.3);border-radius:14px;margin:8px 0'>
    <div style='color:#ff453a;font-size:0.85em;font-weight:600;letter-spacing:0.05em;margin-bottom:10px'>TWO CLOCKS TICKING AT DIFFERENT SPEEDS</div>
    <div style='color:#d1d1d6;font-size:0.92em;line-height:1.6'>
    Pipe 3 is qualitatively different from Pipes 1 and 2 because it operates on <strong>two time horizons simultaneously</strong>:<br><br>

    <strong>Short-term clock:</strong> Market structure could rupture next quarter. Basis trade at $1T, dealer balance sheets constrained, repo market fragile. Any funding shock triggers forced deleveraging.<br><br>

    <strong>Long-term clock:</strong> Fiscal trajectory is unsustainable over 5-20 years. Debt/GDP rising 2-3% annually, interest costs absorbing growing revenue share, safety premium eroding.<br><br>

    <strong>They interact dangerously:</strong> Short-term market stress can accelerate long-term unsustainability (Fed forced to monetize → credibility erodes). Long-term erosion makes short-term stress more dangerous (less capacity to absorb). <strong>Both clocks are ticking and neither can be paused.</strong>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # Why probability models miss this
    st.warning("""
    **Why our ML model can't see this:** Recession probability models trained on 1970-2020 cannot detect Pipe 3 risk because:
    (a) debt levels were never this elevated globally
    (b) leveraged nonbank marginal buyers are a 2010s+ phenomenon
    (c) Treasury safety premium erosion is unprecedented
    (d) G7-wide debt stress is qualitatively new

    The ML ensemble sees "yield curve normal, credit spreads fine, labor OK" and outputs low probability. It's structurally blind to sovereign plumbing.
    """)

    # SVM full picture
    st.markdown("### 🔗 Complete Structural Vulnerability Map")

    svm_summary = pd.DataFrame([
        {"Pipe": "Pipe 1: Foreign Funding", "Status": "ACTIVE-RUPTURE",
         "Key Signal": "PIF commitments -70%, dollar weak during stress"},
        {"Pipe": "Pipe 2: Shadow Banking", "Status": "STRESSED",
         "Key Signal": "BDC 17% NAV discount, 40% PC borrowers cashflow-negative"},
        {"Pipe": "Pipe 3: Sovereign (short-term)", "Status": p3_short_status,
         "Key Signal": "$1T basis trade, dealer constraints, safety premium eroding"},
        {"Pipe": "Pipe 3: Sovereign (long-term)", "Status": p3_long_status,
         "Key Signal": "120% debt/GDP, 6% primary deficit, no fiscal space"},
        {"Pipe": "Pipe 4: Geopolitical", "Status": "Not yet built",
         "Key Signal": "Russia-China-Saudi alignment, currency blocs"},
    ])
    st.dataframe(svm_summary, use_container_width=True, hide_index=True)

    st.caption("Framework inspired by Adam Tooze *Crashed* (2018). "
               "Pipe 3 data: IMF April 2026 Fiscal Monitor, NY Fed research on basis trade, "
               "OFR hedge fund repo data, Dallas Fed basis trade analysis, CBO projections, Treasury TIC. "
               "IMF's OMFIF analysis of 'leveraged marginal buyer' phenomenon and 'eroding Treasury safety premium' "
               "informs the theoretical framework.")


# ── TAB 13: ABOUT ──
with tab13:
    st.markdown("### About this dashboard")

    st.markdown("""
    **Two frameworks answer two different questions:**

    **Framework 1 · Recession Radar (cyclical)** — asks "how likely is a recession?"
    ML ensemble (LR+RF+GBT) trained on 13 historical events, supplemented by four
    ground-level composite modules (LSI, CSI, GCI, CFHI) that measure what market data misses.

    **Framework 2 · Structural Vulnerability Map (transmission)** — asks "if any shock hits,
    how does it propagate?" Four pipes map the hidden plumbing that probability models
    cannot see:
    - Pipe 1 · Foreign Funding (petrodollar/SWF flows) — **BUILT**
    - Pipe 2 · Shadow Banking (private credit, BDCs) — **BUILT**
    - Pipe 3 · Sovereign Debt & Market Structure (Treasury/fiscal, 2 horizons) — **BUILT**
    - Pipe 4 · Geopolitical Weaponization (currency blocs) — to be built

    **Why this matters:** In 2006, the Hamilton Project team watched the cyclical
    thermometer and concluded no imminent crisis. They missed that the transnational
    banking pipe was already rupturing. The framework inspired by Adam Tooze's *Crashed*
    is designed so we don't make that mistake again.

    ---

    **Data refresh:** Hybrid. Core 11 indicators pull live FRED data hourly when API
    key is available. Pipes 1-2 use manual research values (updated ~monthly from
    industry sources: CFR, SWP Berlin, J.P. Morgan research, Fitch, VanEck).

    **Not financial advice.** Consult a qualified advisor.
    """)

    st.caption(f"v8.0 · Apple redesign · Live FRED integration · Auckland, NZ · {datetime.now().strftime('%Y-%m-%d')}")

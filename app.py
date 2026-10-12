"""
FedSpam-Shield — Spam Detection UI (inference only)

Run:   streamlit run app.py

Expects (relative to this file):
    model/global_model.pt
    model/tfidf_vectorizer.joblib
    model/velocity_scaler.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "global_model.pt"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
SCALER_PATH = MODEL_DIR / "velocity_scaler.json"
META_PATH = MODEL_DIR / "global_metadata.json"

THRESHOLD = 0.5

st.set_page_config(page_title="FedSpam-Shield", page_icon="🛡️", layout="wide")


# ──────────────────────────────────────────────────────────────
# Model architecture — must match server.py / clients exactly
# ──────────────────────────────────────────────────────────────
class SpamMLP(nn.Module):
    """Input -> Dense -> ReLU -> Dropout -> Dense(1); outputs logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# ──────────────────────────────────────────────────────────────
# Loading (cached — runs once per server start)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading global model…")
def load_artifacts():
    missing = [p.name for p in (MODEL_PATH, TFIDF_PATH, SCALER_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {MODEL_DIR}: {', '.join(missing)}")

    # Model — dims are read from the saved weights (net.0.weight: [hidden, input])
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    hidden_dim, input_dim = state_dict["net.0.weight"].shape
    model = SpamMLP(input_dim, hidden_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # disables dropout

    meta = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)

    # TF-IDF
    vectorizer = joblib.load(TFIDF_PATH)

    # Velocity scaler
    with open(SCALER_PATH, "r") as f:
        sc = json.load(f)
    def _num(*keys, default):
        for k in keys:
            if k in sc:
                v = sc[k]
                return float(v[0] if isinstance(v, list) else v)
        return default

    mean = _num("mean", "mean_", "mu", default=0.0)
    std = _num("std", "scale", "scale_", "sigma", default=1.0) or 1.0

    return model, vectorizer, mean, std, meta


# ──────────────────────────────────────────────────────────────
# Feature engineering + prediction
# ──────────────────────────────────────────────────────────────
def build_features(subject, body, velocity, header_mismatch, vectorizer, mean, std):
    text = f"{subject} {body}".strip()
    tfidf = vectorizer.transform([text]).toarray().astype(np.float32)  # (1, V)
    vel_norm = (np.log1p(velocity) - mean) / std
    extra = np.array([[vel_norm, float(header_mismatch)]], dtype=np.float32)
    return np.hstack([tfidf, extra])  # (1, V + 2)


@torch.no_grad()
def predict(model, features):
    logit = model(torch.from_numpy(features))
    return float(torch.sigmoid(logit)[0])  # model outputs logits


def confidence_label(p):
    margin = abs(p - THRESHOLD) * 2  # 0 at threshold → 1 at extremes
    if margin >= 0.8:
        return "Very high", margin
    if margin >= 0.5:
        return "High", margin
    if margin >= 0.2:
        return "Moderate", margin
    return "Low", margin


# ──────────────────────────────────────────────────────────────
# UI — "Security console" theme
# ──────────────────────────────────────────────────────────────
C = {
    "bg": "#0B1020", "surface": "#121A2E", "surface2": "#18223A", "border": "#243150",
    "text": "#E6EAF2", "muted": "#8B95AD", "cyan": "#22D3EE", "violet": "#8B5CF6",
    "spam": "#F43F5E", "ham": "#10B981",
}

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

      html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea {{ font-family: 'Inter', sans-serif; }}
      .stApp {{ background: radial-gradient(1200px 600px at 10% -10%, #1B2550 0%, {C['bg']} 55%) fixed; color: {C['text']}; }}
      header[data-testid="stHeader"] {{ background: transparent; }}
      .block-container {{ padding-top: 2rem; max-width: 1200px; }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{ background: {C['surface']}; border-right: 1px solid {C['border']}; }}
      section[data-testid="stSidebar"] * {{ color: {C['text']}; }}

      /* Hero */
      .hero {{ border-radius: 18px; padding: 1.6rem 2rem; margin-bottom: 1.6rem;
               background: linear-gradient(120deg, rgba(34,211,238,.16), rgba(139,92,246,.18));
               border: 1px solid {C['border']}; display:flex; align-items:center; gap:1.2rem; }}
      .hero-icon {{ width:58px; height:58px; border-radius:14px; display:grid; place-items:center; font-size:1.8rem;
                    background: linear-gradient(135deg, {C['cyan']}, {C['violet']}); box-shadow: 0 8px 30px rgba(34,211,238,.25); }}
      .hero h1 {{ margin:0; font-size:2rem; font-weight:800; letter-spacing:-.02em;
                  background: linear-gradient(90deg, #fff, {C['cyan']}); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      .hero p {{ margin:.2rem 0 0; color:{C['muted']}; font-size:.95rem; }}
      .chip {{ display:inline-block; padding:.18rem .6rem; margin-right:.4rem; border-radius:999px; font-size:.72rem; font-weight:600;
               background:{C['surface2']}; border:1px solid {C['border']}; color:{C['cyan']}; }}

      /* Section labels */
      .section-label {{ font-size:.75rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
                        color:{C['muted']}; margin:.2rem 0 .8rem; }}

      /* Cards: the form and bordered containers */
      div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {{
          background:{C['surface']}; border:1px solid {C['border']} !important; border-radius:16px; padding:1.2rem 1.3rem; }}

      /* Inputs */
      .stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {{
          background:{C['bg']} !important; color:{C['text']} !important; border:1px solid {C['border']} !important; border-radius:10px !important; }}
      .stTextInput input:focus, .stTextArea textarea:focus {{ border-color:{C['cyan']} !important; box-shadow:0 0 0 3px rgba(34,211,238,.18) !important; }}
      label, .stSlider label p {{ color:{C['text']} !important; font-weight:600 !important; }}

      /* Primary button */
      .stFormSubmitButton button {{ background: linear-gradient(90deg, {C['cyan']}, {C['violet']}) !important; color:#0B1020 !important;
          border:0 !important; border-radius:12px !important; font-weight:800 !important; padding:.7rem 0 !important; letter-spacing:.02em; }}
      .stFormSubmitButton button p {{ font-weight:800 !important; color:#0B1020 !important; }}
      .stFormSubmitButton button:hover {{ filter:brightness(1.1); box-shadow:0 8px 24px rgba(139,92,246,.35); }}

      /* Verdict */
      .verdict {{ border-radius:16px; padding:1.4rem; text-align:center; margin-bottom:1rem; }}
      .verdict .label {{ font-size:2.2rem; font-weight:800; letter-spacing:.04em; }}
      .verdict .sub {{ color:{C['muted']}; font-size:.9rem; margin-top:.2rem; }}
      .verdict.spam {{ background:rgba(244,63,94,.10); border:1px solid rgba(244,63,94,.55); box-shadow:0 0 40px rgba(244,63,94,.15) inset; }}
      .verdict.spam .label {{ color:{C['spam']}; }}
      .verdict.ham  {{ background:rgba(16,185,129,.10); border:1px solid rgba(16,185,129,.55); box-shadow:0 0 40px rgba(16,185,129,.15) inset; }}
      .verdict.ham .label {{ color:{C['ham']}; }}

      /* Probability ring */
      .ring-wrap {{ display:flex; justify-content:center; margin:.4rem 0 1.2rem; }}
      .ring {{ width:170px; height:170px; border-radius:50%; display:grid; place-items:center; }}
      .ring-inner {{ width:136px; height:136px; border-radius:50%; background:{C['surface']}; display:grid; place-items:center; text-align:center; }}
      .ring-val {{ font-family:'JetBrains Mono', monospace; font-size:1.9rem; font-weight:500; color:{C['text']}; }}
      .ring-cap {{ font-size:.7rem; color:{C['muted']}; letter-spacing:.1em; text-transform:uppercase; }}

      /* Stat tiles */
      .tiles {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.6rem; }}
      .tile {{ background:{C['surface2']}; border:1px solid {C['border']}; border-radius:12px; padding:.75rem .8rem; }}
      .tile .k {{ font-size:.68rem; color:{C['muted']}; text-transform:uppercase; letter-spacing:.1em; }}
      .tile .v {{ font-size:1.15rem; font-weight:700; margin-top:.2rem; color:{C['text']}; }}

      /* Empty state */
      .empty {{ text-align:center; padding:3rem 1rem; color:{C['muted']}; }}
      .empty .big {{ font-size:3rem; opacity:.6; }}

      /* Info rows in sidebar */
      .kv {{ display:flex; justify-content:space-between; padding:.45rem 0; border-bottom:1px dashed {C['border']}; font-size:.85rem; }}
      .kv span:first-child {{ color:{C['muted']}; }}
      .kv span:last-child {{ font-family:'JetBrains Mono', monospace; color:{C['cyan']}; }}

      .footer {{ text-align:center; color:{C['muted']}; font-size:.8rem; margin-top:2rem; padding-top:1rem; border-top:1px solid {C['border']}; }}
      div[data-testid="stExpander"] {{ background:{C['surface2']}; border:1px solid {C['border']}; border-radius:12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-icon">🛡️</div>
      <div>
        <h1>FedSpam-Shield</h1>
        <p>Real-time spam detection powered by a federated-learning global model</p>
        <div style="margin-top:.55rem">
          <span class="chip">FedAvg · Flower</span><span class="chip">SpamMLP</span><span class="chip">Inference only</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, vectorizer, v_mean, v_std, meta = load_artifacts()
except Exception as e:
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

expected_dim = model.net[0].in_features
vocab_dim = len(getattr(vectorizer, "vocabulary_", {})) or None


def kv(k, v):
    return f'<div class="kv"><span>{k}</span><span>{v}</span></div>'


with st.sidebar:
    st.markdown('<div class="section-label">Global model</div>', unsafe_allow_html=True)
    rows = [kv("Architecture", "SpamMLP"), kv("Hidden units", model.net[0].out_features)]
    if meta:
        rows.append(kv("Aggregation", meta.get("aggregation", "FedAvg").split(" ")[0]))
        rows.append(kv("FL round", f"{meta.get('round', '?')}/{meta.get('num_rounds', '?')}"))
    if vocab_dim:
        rows.append(kv("TF-IDF features", f"{vocab_dim:,}"))
    rows.append(kv("Input size", f"{expected_dim:,}"))
    rows.append(kv("Velocity μ", f"{v_mean:.4f}"))
    rows.append(kv("Velocity σ", f"{v_std:.4f}"))
    rows.append(kv("Threshold", THRESHOLD))
    st.markdown("".join(rows), unsafe_allow_html=True)

if vocab_dim and expected_dim != vocab_dim + 2:
    st.warning(
        f"Model expects {expected_dim} features but TF-IDF + 2 gives {vocab_dim + 2}. "
        "Check that the vectorizer matches the trained model."
    )

left, right = st.columns([1.35, 1], gap="large")

with left:
    st.markdown('<div class="section-label">📧 Email input</div>', unsafe_allow_html=True)
    with st.form("email_form", border=False):
        subject = st.text_input("Subject", placeholder="e.g. Congratulations! You've won a prize")
        body = st.text_area("Body", height=220, placeholder="Paste the email body here…")
        c1, c2 = st.columns(2)
        with c1:
            velocity = st.slider("Email velocity", min_value=1, max_value=20, value=1,
                                 help="How many emails this sender sent in a short window.")
        with c2:
            header_mismatch = st.selectbox("Header mismatch", options=[0, 1],
                                           format_func=lambda v: f"{v} — {'Mismatch' if v else 'Match'}",
                                           help="1 if From/Reply-To/Return-Path headers disagree.")
        submitted = st.form_submit_button("ANALYZE EMAIL  →", use_container_width=True, type="primary")

with right:
    st.markdown('<div class="section-label">📊 Analysis</div>', unsafe_allow_html=True)
    panel = st.container(border=True)

    with panel:
        if not submitted:
            st.markdown(
                '<div class="empty"><div class="big">✉️</div>'
                '<div>Enter an email and click <b>Analyze</b><br>to see the verdict here.</div></div>',
                unsafe_allow_html=True,
            )
        elif not subject.strip() and not body.strip():
            st.warning("Please enter a subject or body.")
        else:
            with st.spinner("Analyzing email…"):
                try:
                    feats = build_features(subject, body, velocity, header_mismatch, vectorizer, v_mean, v_std)
                    prob = predict(model, feats)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            is_spam = prob > THRESHOLD
            level, _ = confidence_label(prob)
            color = C["spam"] if is_spam else C["ham"]
            deg = prob * 360

            if is_spam:
                verdict = ('<div class="verdict spam"><div class="label">🚨 SPAM</div>'
                           '<div class="sub">This email looks malicious or unsolicited</div></div>')
            else:
                verdict = ('<div class="verdict ham"><div class="label">✅ HAM</div>'
                           '<div class="sub">This email looks legitimate</div></div>')

            st.markdown(
                verdict
                + f'''<div class="ring-wrap"><div class="ring"
                        style="background:conic-gradient({color} {deg:.1f}deg, {C['border']} {deg:.1f}deg);">
                        <div class="ring-inner"><div>
                          <div class="ring-val">{prob:.1%}</div><div class="ring-cap">spam probability</div>
                        </div></div></div></div>
                    <div class="tiles">
                      <div class="tile"><div class="k">Score</div><div class="v">{prob:.4f}</div></div>
                      <div class="tile"><div class="k">Verdict</div><div class="v" style="color:{color}">{"SPAM" if is_spam else "HAM"}</div></div>
                      <div class="tile"><div class="k">Confidence</div><div class="v">{level}</div></div>
                    </div>''',
                unsafe_allow_html=True,
            )

            st.write("")
            with st.expander("Feature details"):
                st.markdown(
                    kv("Text length", f"{len((subject + ' ' + body).strip())} chars")
                    + kv("Non-zero TF-IDF terms", int((feats[0, :-2] != 0).sum()))
                    + kv("Velocity (scaled)", f"{velocity} → {feats[0, -2]:.4f}")
                    + kv("Header mismatch", header_mismatch)
                    + kv("Feature vector size", feats.shape[1]),
                    unsafe_allow_html=True,
                )

st.markdown(
    '<div class="footer">FedSpam-Shield · Global model aggregated via federated learning · No retraining performed</div>',
    unsafe_allow_html=True,
)

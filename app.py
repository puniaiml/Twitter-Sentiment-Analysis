import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import random
import os
from transformers import AutoTokenizer, BertModel
import plotly.graph_objects as go
from collections import Counter

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentAI · BERT + BiLSTM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07090f;
    color: #e8eaf0;
}

.stApp { background: #07090f; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1300px; }

[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2533;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

.hero-wrap {
    background: linear-gradient(135deg, #0f1724 0%, #0a1628 40%, #111827 100%);
    border: 1px solid #1e2d47;
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(139,92,246,0.10) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #e0f2fe 0%, #38bdf8 40%, #818cf8 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-desc {
    color: #94a3b8;
    font-size: 1rem;
    max-width: 560px;
    line-height: 1.6;
}
.badge-row { display: flex; gap: 0.6rem; margin-top: 1.4rem; flex-wrap: wrap; }
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.3rem 0.75rem;
    border-radius: 999px;
    border: 1px solid;
    letter-spacing: 0.04em;
    font-weight: 500;
}
.badge-blue  { color: #38bdf8; border-color: #1e4d6b; background: rgba(56,189,248,0.07); }
.badge-purple{ color: #a78bfa; border-color: #3b2b6b; background: rgba(167,139,250,0.07); }
.badge-green { color: #4ade80; border-color: #1b4d2e; background: rgba(74,222,128,0.07); }

.card {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2a3a55; }
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #cbd5e1;
    letter-spacing: 0.03em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

textarea {
    background: #111827 !important;
    border: 1px solid #1e2d47 !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    caret-color: #38bdf8 !important;
    resize: vertical !important;
}
textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
    outline: none !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 1.5rem;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(56,189,248,0.15);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(56,189,248,0.25);
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
}
.stButton > button:active { transform: translateY(0); }

.result-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0; }
.sentiment-card {
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.sentiment-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.sent-positive { background: rgba(16,185,129,0.07); border-color: rgba(16,185,129,0.25); }
.sent-positive::before { background: linear-gradient(90deg, #10b981, #34d399); }
.sent-negative { background: rgba(239,68,68,0.07); border-color: rgba(239,68,68,0.25); }
.sent-negative::before { background: linear-gradient(90deg, #ef4444, #f87171); }
.sent-neutral { background: rgba(251,191,36,0.07); border-color: rgba(251,191,36,0.25); }
.sent-neutral::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.sent-irrelevant { background: rgba(148,163,184,0.07); border-color: rgba(148,163,184,0.25); }
.sent-irrelevant::before { background: linear-gradient(90deg, #64748b, #94a3b8); }

.sent-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    opacity: 0.7;
}
.sent-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
}
.sent-bar-track {
    margin-top: 0.6rem;
    height: 4px;
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    overflow: hidden;
}
.sent-bar-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }

.winner-box {
    background: linear-gradient(135deg, #0f2040 0%, #1a1040 100%);
    border: 1px solid #2a3a6a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
}
.winner-emoji { font-size: 3.5rem; margin-bottom: 0.5rem; }
.winner-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.winner-sentiment {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin: 0.3rem 0;
}
.winner-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.88rem;
    color: #38bdf8;
}

.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 800;
    color: #38bdf8;
    line-height: 1;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.4rem;
}

.sidebar-section {
    background: #111827;
    border: 1px solid #1e2533;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
}
.sidebar-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.8rem;
}

.arch-layer {
    background: #111827;
    border: 1px solid #1e2533;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    transition: border-color 0.2s;
}
.arch-layer:hover { border-color: #38bdf8; }
.arch-name { color: #e2e8f0; font-weight: 500; font-family: 'DM Mono', monospace; font-size: 0.8rem; }
.arch-detail { color: #475569; font-size: 0.72rem; font-family: 'DM Mono', monospace; }

.hist-item {
    background: #0d1117;
    border: 1px solid #1e2533;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.85rem;
}
.hist-text { color: #94a3b8; flex: 1; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; margin-right: 1rem; }
.hist-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    white-space: nowrap;
}
.hb-pos { background: rgba(16,185,129,0.15); color: #34d399; }
.hb-neg { background: rgba(239,68,68,0.15); color: #f87171; }
.hb-neu { background: rgba(251,191,36,0.15); color: #fbbf24; }
.hb-irr { background: rgba(148,163,184,0.15); color: #94a3b8; }

hr { border-color: #1e2533 !important; }
.js-plotly-plot .plotly .main-svg { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model Definition ────────────────────────────────────────────────────────
# IMPORTANT: This must match the architecture used during training exactly.
class BERT_LSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2, bidirectional=True):
        super(BERT_LSTM, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        lstm_out_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_out_size, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(input_ids=sent_id, attention_mask=mask,
                           return_dict=False, output_hidden_states=True)
        x = cls_hs[0]
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        x = self.fc(lstm_out[:, -1, :])
        return self.softmax(x)


# ─── Constants ───────────────────────────────────────────────────────────────
LABELS = ["Irrelevant", "Negative", "Neutral", "Positive"]
LABEL_COLORS = {
    "Positive":   ("#10b981", "sent-positive",   "🟢", "hb-pos"),
    "Negative":   ("#ef4444", "sent-negative",   "🔴", "hb-neg"),
    "Neutral":    ("#f59e0b", "sent-neutral",    "🟡", "hb-neu"),
    "Irrelevant": ("#64748b", "sent-irrelevant", "⚪", "hb-irr"),
}
LABEL_BARS = {
    "Positive":   "linear-gradient(90deg,#10b981,#34d399)",
    "Negative":   "linear-gradient(90deg,#ef4444,#f87171)",
    "Neutral":    "linear-gradient(90deg,#f59e0b,#fbbf24)",
    "Irrelevant": "linear-gradient(90deg,#475569,#94a3b8)",
}
HF_REPO_ID = "puneethas26/sentiment-bert-bilstm"   # ← replace with your Hugging Face repo ID

SAMPLE_TWEETS = [
    "I absolutely love this new update! Best thing ever! 🎉",
    "This is the worst product I have ever purchased. Total waste of money.",
    "Just had a meeting about quarterly results. Numbers look okay.",
    "Watching the game tonight, not sure who'll win.",
    "OMG the customer support team is incredible! Fixed my issue instantly! 💯",
    "I can't believe they removed my favorite feature. So disappointed 😤",
    "New album drops next Friday. Haven't heard it yet.",
    "Traffic was bad this morning. Finally made it to work.",
]


# ─── Session State ────────────────────────────────────────────────────────────
if "history"      not in st.session_state: st.session_state.history      = []
if "tweet_input"  not in st.session_state: st.session_state.tweet_input  = ""
if "model_loaded" not in st.session_state: st.session_state.model_loaded = False


# ─── Model Loading ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="best_model.pth"
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = BERT_LSTM(num_classes=4, hidden_size=128, num_layers=2, bidirectional=True)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model, tokenizer


# ─── Inference ───────────────────────────────────────────────────────────────
def predict(text: str, model, tokenizer):
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.exp(logits).squeeze().numpy()

    pred_idx   = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label      = LABELS[pred_idx]
    prob_dict  = {LABELS[i]: float(probs[i]) for i in range(4)}

    return label, confidence, prob_dict


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1.5rem;'>
      <span style='font-family:Syne,sans-serif; font-size:1.15rem; font-weight:800;
                   background:linear-gradient(90deg,#38bdf8,#818cf8);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        SentimentAI
      </span><br>
      <span style='font-family:DM Mono,monospace; font-size:0.65rem; color:#475569; letter-spacing:0.1em;'>
        BERT + BiLSTM · v1.0
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Status ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚡ Model Status</div>', unsafe_allow_html=True)

    if st.button("🚀 Load / Reload Model", key="load_btn"):
        with st.spinner("Loading weights from best_model.pth …"):
            try:
                load_model_and_tokenizer.clear()
                load_model_and_tokenizer()  
                st.session_state.model_loaded = True
                st.success("Model loaded successfully!")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error loading model: {e}")

    # Show whether model file exists
    model_file_exists = True  # file lives on HuggingFace, always reachable

    if st.session_state.model_loaded:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:0.5rem;margin-top:0.7rem;'>
          <span style='width:8px;height:8px;background:#10b981;border-radius:50%;display:inline-block;'></span>
          <span style='font-family:DM Mono,monospace;font-size:0.78rem;color:#34d399;'>ONLINE · best_model.pth</span>
        </div>
        """, unsafe_allow_html=True)
    elif not model_file_exists:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:0.5rem;margin-top:0.7rem;'>
          <span style='width:8px;height:8px;background:#f59e0b;border-radius:50%;display:inline-block;'></span>
          <span style='font-family:DM Mono,monospace;font-size:0.78rem;color:#fbbf24;'>best_model.pth NOT FOUND</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:0.5rem;margin-top:0.7rem;'>
          <span style='width:8px;height:8px;background:#ef4444;border-radius:50%;display:inline-block;'></span>
          <span style='font-family:DM Mono,monospace;font-size:0.78rem;color:#f87171;'>NOT LOADED — click above</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Architecture ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🏗 Architecture</div>', unsafe_allow_html=True)
    for layer, detail in [
        ("BERT Encoder",  "bert-base-uncased"),
        ("Dropout",       "p = 0.10"),
        ("BiLSTM",        "2 layers · hidden=128"),
        ("Linear",        "256 → 4 classes"),
        ("Log-Softmax",   "output activation"),
    ]:
        st.markdown(f"""
        <div class="arch-layer">
          <span class="arch-name">{layer}</span>
          <span class="arch-detail">{detail}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Performance ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📊 Performance</div>', unsafe_allow_html=True)
    for metric, val, col in [
        ("Accuracy",  "90.0%", "#38bdf8"),
        ("Precision", "90.0%", "#a78bfa"),
        ("Recall",    "90.0%", "#34d399"),
        ("F1 Score",  "90.0%", "#f472b6"),
    ]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
                    padding:0.4rem 0;border-bottom:1px solid #1e2533;'>
          <span style='font-family:DM Mono,monospace;font-size:0.78rem;color:#64748b;'>{metric}</span>
          <span style='font-family:Syne,sans-serif;font-size:0.92rem;font-weight:700;color:{col};'>{val}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<div style='height:0.2rem;'></div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick Samples ──
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">💬 Quick Samples</div>', unsafe_allow_html=True)
    for tw in SAMPLE_TWEETS[:5]:
        preview = tw[:40] + ("…" if len(tw) > 40 else "")
        if st.button(f"📝 {preview}", key=f"sample_{tw[:20]}"):
            st.session_state.tweet_input = tw
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Main Layout ─────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-wrap">
  <div class="hero-sub">🧠 Deep Learning · NLP Pipeline</div>
  <div class="hero-title">Twitter Sentiment<br>Intelligence</div>
  <div class="hero-desc">
    Powered by a <strong style="color:#38bdf8;">BERT-base-uncased</strong> encoder
    stacked with a bidirectional LSTM — classifying tweets across
    <strong style="color:#a78bfa;">4 sentiment categories</strong> with ~90% accuracy.
  </div>
  <div class="badge-row">
    <span class="badge badge-blue">BERT-base-uncased</span>
    <span class="badge badge-purple">BiLSTM · 2 Layers</span>
    <span class="badge badge-green">90% Accuracy · 7.2K Test</span>
    <span class="badge badge-blue">4-Class Classification</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Dataset stats
st.markdown("""
<div class="metric-row">
  <div class="metric-card">
    <div class="metric-val">74,681</div>
    <div class="metric-lbl">Training Tweets</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">90.0%</div>
    <div class="metric-lbl">Test Accuracy</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">4</div>
    <div class="metric-lbl">Sentiment Classes</div>
  </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✍️ Enter Tweet</div>', unsafe_allow_html=True)

    tweet_text = st.text_area(
        label="tweet",
        label_visibility="collapsed",
        placeholder="Type or paste a tweet here…",
        value=st.session_state.tweet_input,
        height=130,
        key="tweet_area"
    )

    char_count = len(tweet_text)
    col_cc, col_btn = st.columns([1, 2])
    with col_cc:
        color = "#ef4444" if char_count > 280 else "#38bdf8"
        st.markdown(f"""
        <div style='font-family:DM Mono,monospace;font-size:0.78rem;color:{color};
                    padding-top:0.65rem;'>{char_count}/280</div>
        """, unsafe_allow_html=True)
    with col_btn:
        analyse_btn = st.button("⚡ Analyse Sentiment", key="analyse")

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🎲 Try a Random Sample", key="random"):
        st.session_state.tweet_input = random.choice(SAMPLE_TWEETS)
        st.rerun()

    # ── Run inference ──
    if analyse_btn and tweet_text.strip():
        if not st.session_state.model_loaded:
            st.warning("⚠️  Please load the model first — click **🚀 Load / Reload Model** in the sidebar.")
        else:
            try:
                model, tokenizer = load_model_and_tokenizer()
                with st.spinner("Analysing…"):
                    time.sleep(0.2)
                    label, confidence, probs = predict(tweet_text, model, tokenizer)

                # Save to history
                st.session_state.history.insert(0, {
                    "text":       tweet_text,
                    "label":      label,
                    "confidence": confidence,
                    "probs":      probs,
                })

                st.markdown("---")

                color, css_cls, emoji, _ = LABEL_COLORS[label]
                st.markdown(f"""
                <div class="winner-box">
                  <div class="winner-emoji">{emoji}</div>
                  <div class="winner-label">Predicted Sentiment</div>
                  <div class="winner-sentiment" style="color:{color};">{label}</div>
                  <div class="winner-confidence">Confidence · {confidence*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="result-grid">', unsafe_allow_html=True)
                for lbl in LABELS:
                    pct      = probs[lbl] * 100
                    bar_grad = LABEL_BARS[lbl]
                    col, css, em, _ = LABEL_COLORS[lbl]
                    st.markdown(f"""
                    <div class="sentiment-card {css}">
                      <div class="sent-label">{em} {lbl}</div>
                      <div class="sent-pct" style="color:{col};">{pct:.1f}%</div>
                      <div class="sent-bar-track">
                        <div class="sent-bar-fill" style="width:{pct:.1f}%;background:{bar_grad};"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference error: {e}")

    elif analyse_btn:
        st.warning("Please enter some tweet text first.")

with col_right:
    if st.session_state.history:
        latest = st.session_state.history[0]
        probs  = latest["probs"]

        # Radar chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📡 Probability Radar</div>', unsafe_allow_html=True)

        categories = list(probs.keys())
        values     = [probs[k] * 100 for k in categories]
        values_c   = values + [values[0]]
        cats_c     = categories + [categories[0]]

        fig_radar = go.Figure(go.Scatterpolar(
            r=values_c, theta=cats_c,
            fill='toself',
            fillcolor='rgba(56,189,248,0.12)',
            line=dict(color='#38bdf8', width=2),
            marker=dict(color='#38bdf8', size=6),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(visible=True, range=[0, 100],
                                tickfont=dict(color='#475569', size=10, family='DM Mono'),
                                gridcolor='#1e2533', linecolor='#1e2533'),
                angularaxis=dict(tickfont=dict(color='#94a3b8', size=11, family='Syne'),
                                 gridcolor='#1e2533', linecolor='#1e2533'),
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, b=10, l=40, r=40),
            height=270,
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Bar chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Score Breakdown</div>', unsafe_allow_html=True)

        bar_colors = [LABEL_COLORS[k][0] for k in categories]
        fig_bar = go.Figure(go.Bar(
            x=categories, y=values,
            marker=dict(color=bar_colors, opacity=0.85,
                        line=dict(color='rgba(0,0,0,0)', width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=11, family='DM Mono'),
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=20, b=20, l=10, r=10),
            height=230,
            yaxis=dict(visible=False, range=[0, max(values) * 1.25]),
            xaxis=dict(tickfont=dict(color='#94a3b8', size=11, family='Syne'),
                       gridcolor='#1e2533', linecolor='#1e2533'),
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # How it works placeholder
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">ℹ️ How It Works</div>', unsafe_allow_html=True)
        for num, title, desc in [
            ("1", "Tokenize",  "BERT WordPiece tokenizer converts text → token IDs (max 128)"),
            ("2", "Encode",    "BERT extracts contextual embeddings from all layers"),
            ("3", "Sequence",  "BiLSTM reads the sequence forward + backward simultaneously"),
            ("4", "Classify",  "Linear layer maps BiLSTM output → 4 sentiment logits"),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:1rem;margin-bottom:1rem;align-items:flex-start;'>
              <div style='min-width:28px;height:28px;background:rgba(56,189,248,0.15);
                          border:1px solid rgba(56,189,248,0.3);border-radius:50%;
                          display:flex;align-items:center;justify-content:center;
                          font-family:Syne,sans-serif;font-weight:800;font-size:0.78rem;
                          color:#38bdf8;'>{num}</div>
              <div>
                <div style='font-family:Syne,sans-serif;font-size:0.9rem;font-weight:700;
                             color:#e2e8f0;margin-bottom:0.2rem;'>{title}</div>
                <div style='font-family:DM Sans,sans-serif;font-size:0.82rem;color:#64748b;
                             line-height:1.5;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:DM Mono,monospace;font-size:0.78rem;color:#475569;
                    text-align:center;padding-top:0.5rem;'>
          Load the model → Enter a tweet → Click Analyse
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Class legend
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🏷️ Sentiment Classes</div>', unsafe_allow_html=True)
        for lbl, (em, col, desc) in {
            "Positive":   ("🟢", "#10b981", "Happiness, excitement, satisfaction, praise"),
            "Negative":   ("🔴", "#ef4444", "Anger, sadness, frustration, criticism"),
            "Neutral":    ("🟡", "#f59e0b", "Factual, informational, no strong emotion"),
            "Irrelevant": ("⚪", "#64748b", "Unrelated, spam, or off-topic content"),
        }.items():
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;padding:0.6rem 0;
                        border-bottom:1px solid #1e2533;'>
              <span style='font-size:1.1rem;'>{em}</span>
              <div>
                <div style='font-family:Syne,sans-serif;font-size:0.88rem;font-weight:700;
                             color:{col};'>{lbl}</div>
                <div style='font-family:DM Sans,sans-serif;font-size:0.78rem;color:#475569;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─── History Section ─────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;
                color:#e2e8f0;margin-bottom:1rem;'>
      🕓 Analysis History
    </div>
    """, unsafe_allow_html=True)

    hist_col1, hist_col2 = st.columns(2, gap="medium")
    for i, item in enumerate(st.session_state.history[:10]):
        _, _, _, badge_cls = LABEL_COLORS[item["label"]]
        color, _, em, _   = LABEL_COLORS[item["label"]]
        preview = item["text"][:65] + ("…" if len(item["text"]) > 65 else "")
        html = f"""
        <div class="hist-item">
          <div class="hist-text">{preview}</div>
          <div class="hist-badge {badge_cls}">{em} {item['label']} · {item['confidence']*100:.0f}%</div>
        </div>
        """
        (hist_col1 if i % 2 == 0 else hist_col2).markdown(html, unsafe_allow_html=True)

    if len(st.session_state.history) >= 3:
        st.markdown('<div class="card" style="margin-top:1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Session Distribution</div>', unsafe_allow_html=True)
        counts = Counter(h["label"] for h in st.session_state.history)
        fig_pie = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            marker=dict(colors=[LABEL_COLORS[k][0] for k in counts.keys()],
                        line=dict(color='#07090f', width=3)),
            textfont=dict(family='DM Mono', size=12, color='#e2e8f0'),
            hole=0.55,
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(font=dict(color='#94a3b8', family='DM Mono', size=11),
                        bgcolor='rgba(0,0,0,0)'),
            margin=dict(t=10, b=10, l=10, r=10), height=260,
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;font-family:DM Mono,monospace;font-size:0.72rem;
            color:#334155;padding:0.5rem 0 1rem;'>
  SentimentAI · BERT + BiLSTM · Twitter Sentiment Analysis · 4-Class · ~90% Accuracy
</div>
            


""", unsafe_allow_html=True)


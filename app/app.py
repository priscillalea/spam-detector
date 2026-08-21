"""
app.py — Spam Detector — Streamlit interface.

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import time

import streamlit as st

# Allow import from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spam_detector.predict import SpamDetector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* ── Layout tokens ── */
  :root {
    --bg-primary: #0f1117;
    --bg-card: #1a1d27;
    --bg-card-hover: #1f2235;
    --border: #2a2d3e;
    --border-accent: #3d4166;
    --text-primary: #e8eaf6;
    --text-secondary: #8b92b8;
    --text-muted: #565d8a;
    --spam-color: #ff4d6d;
    --spam-bg: rgba(255, 77, 109, 0.08);
    --spam-border: rgba(255, 77, 109, 0.25);
    --ham-color: #4ade80;
    --ham-bg: rgba(74, 222, 128, 0.08);
    --ham-border: rgba(74, 222, 128, 0.25);
    --accent: #6c63ff;
    --accent-hover: #7c73ff;
  }

  /* ── Global background ── */
  .stApp {
    background: var(--bg-primary);
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem; max-width: 1200px; }

  /* ── Header ── */
  .app-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.25rem;
  }
  .app-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
  }
  .app-subtitle {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin: 0;
  }
  .badge-tech {
    display: inline-block;
    background: rgba(108, 99, 255, 0.12);
    border: 1px solid rgba(108, 99, 255, 0.3);
    color: #9d96ff;
    font-size: 0.7rem;
    font-weight: 500;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .divider {
    height: 1px;
    background: var(--border);
    margin: 1.25rem 0;
  }

  /* ── Email compose card ── */
  .compose-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .compose-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }

  /* Override Streamlit textarea */
  .stTextArea textarea {
    background: #12151e !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 0.9rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
  }
  .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108, 99, 255, 0.15) !important;
  }

  /* ── Analyze button ── */
  .stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.625rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.01em !important;
  }
  .stButton > button:hover {
    background: var(--accent-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(108, 99, 255, 0.35) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }

  /* ── Result card ── */
  .result-card {
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid;
  }
  .result-card.spam {
    background: var(--spam-bg);
    border-color: var(--spam-border);
  }
  .result-card.ham {
    background: var(--ham-bg);
    border-color: var(--ham-border);
  }
  .result-label {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  .result-label.spam { color: var(--spam-color); }
  .result-label.ham { color: var(--ham-color); }
  .result-description {
    font-size: 0.825rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
  }

  /* ── Probability bar ── */
  .prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1rem;
  }
  .prob-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    min-width: 80px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
  }
  .prob-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
  }
  .prob-bar-fill.spam { background: var(--spam-color); }
  .prob-bar-fill.ham { background: var(--ham-color); }
  .prob-value {
    font-size: 0.825rem;
    font-weight: 600;
    min-width: 40px;
    text-align: right;
  }
  .prob-value.spam { color: var(--spam-color); }
  .prob-value.ham { color: var(--ham-color); }

  /* ── Probability disclaimer ── */
  .prob-disclaimer {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
    line-height: 1.5;
    font-style: italic;
  }

  /* ── Token tags ── */
  .section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    margin-top: 1.25rem;
  }
  .token-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
  }
  .token-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid;
  }
  .token-tag.spam {
    background: var(--spam-bg);
    border-color: var(--spam-border);
    color: #ff8fab;
  }
  .token-tag.ham {
    background: var(--ham-bg);
    border-color: var(--ham-border);
    color: #86efac;
  }

  /* ── History panel ── */
  .history-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.625rem 0.875rem;
    border-radius: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .history-item:hover { border-color: var(--border-accent); }
  .history-excerpt {
    font-size: 0.8rem;
    color: var(--text-secondary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 170px;
  }
  .history-badge {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .history-badge.spam {
    background: var(--spam-bg);
    color: var(--spam-color);
    border: 1px solid var(--spam-border);
  }
  .history-badge.ham {
    background: var(--ham-bg);
    color: var(--ham-color);
    border: 1px solid var(--ham-border);
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
  }
  .empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
  .empty-text { font-size: 0.875rem; line-height: 1.6; }

  /* ── Info card ── */
  .info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
  }
  .info-card-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
  }
  .info-card-value {
    font-size: 0.875rem;
    color: var(--text-secondary);
  }
  .info-card-value strong {
    color: var(--text-primary);
  }

  /* ── Samples ── */
  .sample-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.4rem 0.75rem;
    color: var(--text-secondary);
    font-size: 0.78rem;
    cursor: pointer;
    display: inline-block;
    margin: 0.25rem 0.25rem 0.25rem 0;
  }
</style>
""", unsafe_allow_html=True)


# ── Model loading (once per session) ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector() -> SpamDetector | None:
    try:
        return SpamDetector()
    except FileNotFoundError:
        return None


detector = load_detector()

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "current_text" not in st.session_state:
    st.session_state.current_text = ""
if "textarea_key" not in st.session_state:
    st.session_state.textarea_key = 0


def _load_sample(text: str) -> None:
    """Callback: load a sample email into the textarea."""
    st.session_state.current_text = text
    st.session_state.textarea_key += 1  # force re-render

# ── Sample emails ─────────────────────────────────────────────────────────────
SAMPLE_SPAM = (
    "Subject: URGENT — You've Been Selected!\n\n"
    "Congratulations! You have been chosen as our lucky winner for a FREE prize "
    "worth $10,000! This is a limited-time offer. Click the link below to claim "
    "your reward immediately. Call us at 1-800-WIN-FREE. Act now before it expires!"
)
SAMPLE_HAM = (
    "Subject: Q3 Report — Action Required\n\n"
    "Hi team,\n\nPlease find the Q3 financial report attached for your review. "
    "We'll discuss the results in Thursday's board meeting. "
    "Let me know if you have any questions beforehand.\n\nBest regards, Sarah"
)
SAMPLE_PHISHING = (
    "Subject: Your account has been compromised\n\n"
    "ALERT: Suspicious activity detected on your account. "
    "Log in immediately at http://secure-bank-login.xyz/verify to prevent "
    "permanent suspension. Failure to act within 24 hours will result in "
    "account closure."
)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div>
    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
      <p class="app-title">Spam Detector</p>
      <span class="badge-tech">TF-IDF · LinearSVC · NLP</span>
    </div>
    <p class="app-subtitle">
      Email classification with classical NLP — trained on the Enron Spam Dataset
    </p>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # ── Model missing warning ─────────────────────────────────────────────────
    if detector is None:
        st.error("Model not found. Run `python -m spam_detector.train` first.")
        st.stop()

    # ── Quick samples ─────────────────────────────────────────────────────────
    st.markdown('<p class="compose-label">Quick Examples</p>', unsafe_allow_html=True)

    sample_col1, sample_col2, sample_col3 = st.columns(3)
    with sample_col1:
        if st.button("Spam sample", key="btn_spam_sample", use_container_width=True,
                     on_click=_load_sample, args=(SAMPLE_SPAM,)):
            pass
    with sample_col2:
        if st.button("Legitimate email", key="btn_ham_sample", use_container_width=True,
                     on_click=_load_sample, args=(SAMPLE_HAM,)):
            pass
    with sample_col3:
        if st.button("Phishing attempt", key="btn_phishing_sample", use_container_width=True,
                     on_click=_load_sample, args=(SAMPLE_PHISHING,)):
            pass

    st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)

    # ── Compose area ──────────────────────────────────────────────────────────
    st.markdown('<p class="compose-label">Email Content</p>', unsafe_allow_html=True)

    email_text = st.text_area(
        label="Email Content",
        value=st.session_state.current_text,
        placeholder="Paste or type an email here…\n\nInclude the subject line for better results.",
        height=240,
        label_visibility="collapsed",
        key=f"email_input_{st.session_state.textarea_key}",
    )

    analyze_clicked = st.button("Analyze Email", key="btn_analyze")

    # ── Result ────────────────────────────────────────────────────────────────
    if analyze_clicked:
        if not email_text.strip():
            st.warning("Please enter some email text before analyzing.")
        else:
            with st.spinner("Analyzing…"):
                result = detector.predict(email_text)
                time.sleep(0.2)  # brief delay so the spinner is visible

            # Add to history
            st.session_state.history.insert(
                0,
                {
                    "text": email_text,
                    "label": result.label,
                    "probability": result.spam_probability,
                }
            )
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]

            cls = "spam" if result.is_spam else "ham"
            label_text = "SPAM" if result.is_spam else "LEGITIMATE"
            desc = (
                "This message shows characteristics commonly associated with spam email."
                if result.is_spam
                else "This message appears to be a legitimate email."
            )

            prob_pct = result.spam_probability * 100
            ham_pct = (1 - result.spam_probability) * 100

            # Result card HTML
            st.markdown(f"""
<div class="result-card {cls}" id="result-card">
  <div class="result-label {cls}">{label_text}</div>
  <p class="result-description">{desc}</p>

  <div class="prob-row">
    <span class="prob-label">Spam score</span>
    <div class="prob-bar-bg">
      <div class="prob-bar-fill spam" style="width:{prob_pct:.1f}%"></div>
    </div>
    <span class="prob-value spam">{prob_pct:.1f}%</span>
  </div>
  <div class="prob-row">
    <span class="prob-label">Ham score</span>
    <div class="prob-bar-bg">
      <div class="prob-bar-fill ham" style="width:{ham_pct:.1f}%"></div>
    </div>
    <span class="prob-value ham">{ham_pct:.1f}%</span>
  </div>
  <p class="prob-disclaimer">
    Scores are calibrated probabilities from a LinearSVC wrapped in
    CalibratedClassifierCV. They reflect model confidence, not ground-truth
    certainty, and may not be perfectly calibrated for all inputs.
  </p>
</div>
""", unsafe_allow_html=True)

            # ── Token explanations ─────────────────────────────────────────────
            if result.top_spam_tokens or result.top_ham_tokens:
                st.markdown(
                    '<p class="section-title">Key signals found in this message</p>',
                    unsafe_allow_html=True,
                )

                tok_col1, tok_col2 = st.columns(2)

                with tok_col1:
                    st.markdown(
                        '<p style="font-size:0.75rem;color:#ff8fab;font-weight:600;'
                        'margin-bottom:0.5rem;">↑ Spam indicators</p>',
                        unsafe_allow_html=True,
                    )
                    if result.top_spam_tokens:
                        tags = " ".join(
                            f'<span class="token-tag spam">{tok}</span>'
                            for tok, _ in result.top_spam_tokens[:8]
                        )
                        st.markdown(
                            f'<div class="token-grid">{tags}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<span style="font-size:0.8rem;color:#565d8a">'
                            'No strong spam signals.</span>',
                            unsafe_allow_html=True,
                        )

                with tok_col2:
                    st.markdown(
                        '<p style="font-size:0.75rem;color:#86efac;font-weight:600;'
                        'margin-bottom:0.5rem;">↓ Legitimate indicators</p>',
                        unsafe_allow_html=True,
                    )
                    if result.top_ham_tokens:
                        tags = " ".join(
                            f'<span class="token-tag ham">{tok}</span>'
                            for tok, _ in result.top_ham_tokens[:8]
                        )
                        st.markdown(
                            f'<div class="token-grid">{tags}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<span style="font-size:0.8rem;color:#565d8a">'
                            'No strong ham signals.</span>',
                            unsafe_allow_html=True,
                        )

                st.markdown("""
<p style="font-size:0.72rem;color:#565d8a;margin-top:0.75rem;font-style:italic;">
  Tokens are derived from the model's decision boundary (LinearSVC coefficients).
  Only tokens present in this specific message are shown — not global feature importance.
</p>
""", unsafe_allow_html=True)

    elif not analyze_clicked and not st.session_state.history:
        # Empty state prompt
        st.markdown("""
<div class="empty-state">
  <p class="empty-text">
    Paste an email above and click <strong>Analyze Email</strong><br>
    to see the classification result.
  </p>
</div>
""", unsafe_allow_html=True)


with right_col:
    # ── Model info ────────────────────────────────────────────────────────────
    st.markdown('<p class="compose-label">Model</p>', unsafe_allow_html=True)

    st.markdown("""
<div class="info-card">
  <div class="info-card-title">Architecture</div>
  <div class="info-card-value">
    <strong>TF-IDF</strong> (unigrams + bigrams, sublinear TF)<br>
    <strong>LinearSVC</strong> + CalibratedClassifierCV
  </div>
</div>
<div class="info-card">
  <div class="info-card-title">Training</div>
  <div class="info-card-value">
    Enron Spam Dataset · 5,171 emails<br>
    4-model comparison, 5-fold cross-validation
  </div>
</div>
<div class="info-card">
  <div class="info-card-title">Test-Set Performance</div>
  <div class="info-card-value">
    F1-spam <strong>0.984</strong> ·
    Recall-spam <strong>0.995</strong> ·
    ROC-AUC <strong>0.999</strong>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── History ───────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="divider"></div>'
        '<p class="compose-label">Analysis History</p>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.markdown(
            '<p style="font-size:0.8rem;color:#565d8a;text-align:center;'
            'padding:1rem 0">No analyses yet.</p>',
            unsafe_allow_html=True,
        )
    else:
        for i, item in enumerate(st.session_state.history):
            excerpt = item["text"].replace("\n", " ")[:45]
            if len(item["text"]) > 45:
                excerpt += "…"
            cls = item["label"]
            badge_text = "SPAM" if cls == "spam" else "HAM"
            prob = item["probability"] * 100

            st.markdown(f"""
<div class="history-item">
  <div>
    <div class="history-excerpt">{excerpt}</div>
    <div style="font-size:0.7rem;color:#565d8a;margin-top:0.2rem;">
      Score: {prob:.0f}%
    </div>
  </div>
  <span class="history-badge {cls}">{badge_text}</span>
</div>
""", unsafe_allow_html=True)

        if st.button("Clear history", key="btn_clear_history"):
            st.session_state.history = []
            st.rerun()

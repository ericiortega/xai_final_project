import sys
import os
import pandas as pd
import altair as alt

# Add parent directory (project root) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from models.bert_model import positivity_score, predict_label
from models.shap_explainer import get_token_importances


# -------------------------------------------------------------------
# THEME / CSS
# -------------------------------------------------------------------


def inject_css():
    st.markdown(
        """
        <style>

        /* -----------------------------------------------------------
           GLOBAL — FONT + BACKGROUND
        ----------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left,
                         #0e1625 0%,
                         #08101d 40%,
                         #020617 80%);
            color: #e5e7eb;
        }

        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1250px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* -----------------------------------------------------------
           HEADER (HERO BAR)
        ----------------------------------------------------------- */
        .app-header-shell {
            background: linear-gradient(135deg,
                        rgba(59,130,246,0.35),
                        rgba(15,23,42,0.98));
            border-radius: 24px;
            border: 1px solid rgba(148,163,184,0.55);
            padding: 14px 20px;
            box-shadow: 0 18px 45px rgba(0,0,0,0.65);
            margin-bottom: 1.6rem;
        }

        .app-header {
            display:flex;
            justify-content:space-between;
            align-items:center;
        }

        .app-header-left {
            display:flex;
            align-items:center;
            gap:1rem;
        }

        .app-header-right {
            display:flex;
            align-items:center;
            gap:0.6rem;
        }

        .logo-pill {
            width: 42px;
            height: 42px;
            border-radius: 999px;
            background: radial-gradient(circle at 20% 0%,
                        #7dd3fc,
                        #3b82f6 50%,
                        #6366f1);
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }

        .badge-chip {
            padding: 6px 14px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.45);
            background: rgba(15,23,42,0.85);
            display:flex;
            align-items:center;
            gap:7px;
            font-size:0.8rem;
        }

        .badge-dot {
            width:7px;
            height:7px;
            background:#22c55e;
            border-radius:999px;
            box-shadow:0 0 10px rgba(34,197,94,0.8);
        }


        /* -----------------------------------------------------------
           CARDS — GLASS EFFECT
        ----------------------------------------------------------- */
        .glass-card {
            background: rgba(15,23,42,0.60);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.10);
            padding: 20px 24px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.55);
            transition: transform 0.15s ease-out,
                        box-shadow 0.15s ease-out;
        }

        .glass-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 40px rgba(0,0,0,0.6);
        }

        .main-input-card { margin-bottom: 1rem; }
        .analysis-card   { margin-top: 0.35rem; }
        .side-card       { margin-bottom: 1rem; }

        .card-section-title {
            font-weight:600;
            font-size:0.9rem;
            margin-bottom:0.75rem;
            display:flex;
            gap:10px;
            align-items:center;
        }

        .card-section-title-icon {
            width:26px;
            height:26px;
            border-radius:999px;
            background:rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.15);
            display:flex;
            align-items:center;
            justify-content:center;
        }

        .card-section-body {
            font-size:0.82rem;
            color:#cbd5e1;
            line-height:1.4;
        }


        /* -----------------------------------------------------------
           TEXT INPUT
        ----------------------------------------------------------- */
        textarea {
            background: rgba(10,15,25,0.9) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            color: #e5e7eb !important;
            font-size:0.95rem !important;
        }

        textarea:focus {
            border-color: #3b82f6 !important;
            box-shadow: 0 0 0 1px #3b82f6 !important;
        }


        /* -----------------------------------------------------------
           ANALYZE BUTTON (PRIMARY)
        ----------------------------------------------------------- */
        .analyze-button .stButton > button {
            background: linear-gradient(90deg, #2563eb, #4f46e5, #7c3aed) !important;
            border: none !important;
            padding: 0.6rem 2rem !important;
            border-radius: 999px !important;
            color: white !important;
            font-weight: 500 !important;
            box-shadow: 0 14px 38px rgba(0,0,0,0.55);
            transition: all 0.15s ease-out;
        }

        .analyze-button .stButton > button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 20px 45px rgba(0,0,0,0.65);
            filter:brightness(1.05);
        }

        .analyze-button .stButton > button:disabled {
            background: rgba(15,23,42,0.75) !important;
            color: #6b7280 !important;
            border: 1px solid rgba(75,85,99,0.8) !important;
            box-shadow:none !important;
        }


        /* -----------------------------------------------------------
           EXAMPLE BUTTONS (NO WHITE)
        ----------------------------------------------------------- */
        .example-btn .stButton > button {
            width: 100%;
            text-align:left;
            white-space:normal!important;

            background: rgba(30,41,59,0.75) !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 16px !important;
            color: #e5e7eb !important;

            padding: 0.55rem 0.9rem !important;
            font-size:0.82rem !important;

            box-shadow: 0 10px 28px rgba(0,0,0,0.55);
            transition: transform 0.15s ease-out,
                        box-shadow 0.15s ease-out,
                        border-color 0.15s ease-out;
        }

        .example-btn .stButton > button:hover {
            background: rgba(59,130,246,0.25) !important;
            border-color: rgba(129,140,248,0.9) !important;
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.65);
        }


        /* -----------------------------------------------------------
           SENTIMENT LABELS
        ----------------------------------------------------------- */
        .sentiment-pill {
            padding:5px 15px;
            border-radius:999px;
            font-size:0.85rem;
            font-weight:500;
        }

        .sentiment-positive {
            background:rgba(34,197,94,0.20);
            border:1px solid #22c55e;
            color:#bbf7d0;
        }

        .sentiment-negative {
            background:rgba(239,68,68,0.20);
            border:1px solid #ef4444;
            color:#fecaca;
        }

        .sentiment-neutral {
            background:rgba(156,163,175,0.20);
            border:1px solid #9ca3af;
            color:#e5e7eb;
        }


        /* -----------------------------------------------------------
           MINI STATS (POS / NEG / NEUTRAL COUNTS)
        ----------------------------------------------------------- */
        .mini-stat-row {
            display:flex;
            gap:10px;
            margin-top:1rem;
        }

        .mini-stat {
            flex:1;
            padding:10px 12px;
            border-radius:18px;
            background:rgba(15,23,42,0.98);
            border:1px solid rgba(148,163,184,0.45);
            font-size:0.8rem;
        }

        .mini-label {
            color:#9ca3af;
            margin-bottom:3px;
        }

        .mini-value {
            font-size:0.95rem;
            font-weight:500;
        }


        /* -----------------------------------------------------------
           SHAP TOKEN PILL
        ----------------------------------------------------------- */
        .token-pill {
            color:white !important;
            font-size:0.84rem !important;
            padding:4px 8px;
            border-radius:999px;
            border:1px solid rgba(255,255,255,0.18);
        }


        /* -----------------------------------------------------------
           GLOBAL BUTTON OVERRIDE (kills ALL white Streamlit buttons)
        ----------------------------------------------------------- */
        button[kind="secondary"] {
            background: rgba(24,31,47,0.85) !important;
            border: 1px solid rgba(148,163,184,0.4) !important;
            color: #e5e7eb !important;

            border-radius: 999px !important;
            padding: 0.5rem 1.4rem !important;
            font-size: 0.82rem !important;

            box-shadow: 0 10px 26px rgba(0,0,0,0.45);
            transition: all 0.15s ease-out;
        }

        button[kind="secondary"]:hover:not(:disabled) {
            background: rgba(59,130,246,0.22) !important;
            border-color: rgba(129,140,248,0.9) !important;
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.6);
        }

        button[kind="secondary"]:disabled {
            background: rgba(15,23,42,0.9) !important;
            color: #6b7280 !important;
            border-color: rgba(55,65,81,0.8) !important;
            box-shadow:none !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# SHAP HELPERS
# -------------------------------------------------------------------


def value_to_color(v, max_abs):
    if max_abs <= 0:
        return "transparent"
    v = max(-max_abs, min(max_abs, v))
    alpha = 0.35 + 0.65 * (abs(v) / max_abs)
    if v > 0:
        return f"rgba(34,197,94,{alpha})"
    else:
        return f"rgba(239,68,68,{alpha})"


def tokens_to_html(tokens, shap_values):
    if not tokens:
        return ""
    max_abs = max(abs(v) for v in shap_values) or 1
    spans = []
    for tok, val in zip(tokens, shap_values):
        bg = value_to_color(val, max_abs)
        spans.append(
            f"<span class='token-pill' style='background-color:{bg}; margin:3px;'>{tok}</span>"
        )
    return " ".join(spans)


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------


def main():

    st.set_page_config(
        page_title="Explainable Sentiment Analyzer",
        page_icon="✨",
        layout="wide",
    )
    inject_css()

    if "text_input" not in st.session_state:
        st.session_state["text_input"] = "I love this"

    # ------------------------------- HEADER ---------------------------------
    st.markdown(
        """
        <div class="app-header-shell">
          <div class="app-header">
            <div class="app-header-left">
              <div class="logo-pill">☻</div>
              <div>
                <div class="app-title-main" style="font-weight:600; font-size:1.1rem;">
                  Explainable Sentiment Analyzer
                </div>
                <div class="app-title-sub" style="font-size:0.82rem; color:#e5e7eb;">
                  AI-powered interpretable sentiment analysis
                </div>
              </div>
            </div>
            <div class="app-header-right">
              <div class="badge-chip">
                <div class="badge-dot"></div>
                <span>BERT + SHAP</span>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_main, col_side = st.columns([2.2, 0.8])

    # ===========================
    # RIGHT PANEL
    # ===========================
    with col_side:

        st.markdown('<div class="glass-card side-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-section-title">
                <div class="card-section-title-icon">ℹ️</div>
                <span>How it works</span>
            </div>
            <div class="card-section-body">
                This app uses a pretrained <b>BERT</b> model to classify sentiment and
                <b>SHAP</b> to explain which words push predictions positive or negative.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Examples
        st.markdown('<div class="glass-card side-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-section-title">
                <div class="card-section-title-icon">✨</div>
                <span>Try an example</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        examples = [
            "I love the design, but the battery life is terrible.",
            "This product exceeded all my expectations. Absolutely amazing.",
            "The customer service was disappointing and the delivery took too long.",
            "It is okay, nothing special but it gets the job done.",
        ]

        for i, ex in enumerate(examples):
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(ex, key=f"example_{i}"):
                st.session_state["text_input"] = ex
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Features
        st.markdown('<div class="glass-card side-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-section-title">
                <div class="card-section-title-icon">⚡</div>
                <span>Features</span>
            </div>
            <div class="card-section-body">
                • Explainable AI with SHAP<br>
                • Real-time sentiment prediction<br>
                • Word-level influence visualization
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ===========================
    # LEFT SIDE — INPUT + ANALYSIS
    # ===========================
    with col_main:

        # INPUT CARD
        st.markdown('<div class="glass-card main-input-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-section-title">
                <div class="card-section-title-icon">📄</div>
                <span>Enter your text</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        text = st.text_area("", key="text_input", height=170)

        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Sentiment")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:0.75rem; color:#9ca3af;'>Press Analyze Sentiment to run the model.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # WHEN ANALYSIS RUNS
        if analyze_clicked and text.strip():

            score, p_pos = positivity_score(text)
            label, _ = predict_label(text)
            confidence = max(p_pos, 1 - p_pos)
            conf_pct = int(confidence * 100)

            if score > 60:
                sentiment = "Positive"
                pill = "sentiment-positive"
            elif score < 40:
                sentiment = "Negative"
                pill = "sentiment-negative"
            else:
                sentiment = "Neutral"
                pill = "sentiment-neutral"

            with st.spinner("Computing SHAP explanations..."):
                tokens, shap_values, base = get_token_importances(text)

            pos_count = sum(v > 0.01 for v in shap_values)
            neg_count = sum(v < -0.01 for v in shap_values)
            neut_count = len(tokens) - pos_count - neg_count

            st.markdown(
                '<div class="glass-card analysis-card">', unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="sentiment-pill {pill}">{sentiment}</div>
                    <div style="font-size:1.4rem; font-weight:600;">{conf_pct}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Mini stats
            st.markdown(
                f"""
                <div class="mini-stat-row">
                    <div class="mini-stat">
                        <div class="mini-label">Positive words</div>
                        <div class="mini-value">{pos_count}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Negative words</div>
                        <div class="mini-value">{neg_count}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Neutral words</div>
                        <div class="mini-value">{neut_count}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence Bar
            st.markdown(
                f"""
                <div style="margin-top:1rem;">
                    <div style="font-size:0.8rem; color:#9ca3af;">Model confidence</div>
                    <div style="height:10px; border-radius:999px; background:rgba(15,23,42,0.8); border:1px solid rgba(75,85,99,0.8);">
                        <div style="height:10px; width:{conf_pct}%; background:linear-gradient(90deg,#22c55e,#4ade80,#22d3ee); border-radius:999px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # SHAP tokens
            st.markdown(
                "<div style='margin-top:1.3rem; font-size:0.85rem; font-weight:600;'>Word-level explanation (SHAP)</div>",
                unsafe_allow_html=True,
            )

            st.markdown(tokens_to_html(tokens, shap_values), unsafe_allow_html=True)

            # SHAP legend
            st.markdown(
                """
                <div style="margin-top:10px; font-size:0.75rem;">
                    <span style="background:rgba(34,197,94,0.22); border:1px solid #22c55e; padding:3px 8px; border-radius:999px; margin-right:8px;">
                        Positive contribution
                    </span>
                    <span style="background:rgba(239,68,68,0.22); border:1px solid #ef4444; padding:3px 8px; border-radius:999px;">
                        Negative contribution
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Bar Chart
            st.markdown(
                "<div style='margin-top:1.2rem; font-weight:600;'>Most influential words</div>",
                unsafe_allow_html=True,
            )

            df = pd.DataFrame({"token": tokens, "shap_value": shap_values})
            df["abs"] = df["shap_value"].abs()
            df = df.sort_values("abs", ascending=False).head(15)

            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "shap_value:Q", title="SHAP impact", scale=alt.Scale(zero=True)
                    ),
                    y=alt.Y(
                        "token:N", sort=alt.SortField(field="abs", order="descending")
                    ),
                    color=alt.condition(
                        "datum.shap_value > 0",
                        alt.value("#22c55e"),
                        alt.value("#ef4444"),
                    ),
                    tooltip=["token", "shap_value"],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # FOOTER
    st.markdown(
        """
        <div style="margin-top:2rem; color:#6b7280; font-size:0.8rem;">
            Built by <b>Eric Ortega</b> &amp; <b>Diya Mirji</b> · For AIPI 590: Emerging Trends in Explainable AI
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

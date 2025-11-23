# app.py
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from models.bert_model import positivity_score, predict_label
from models.shap_explainer import get_token_importances
## theme
def inject_css():
    st.markdown(
        """
        <style>

        /* ==========================================================
           GLOBAL BUTTON OVERRIDE — FIX ALL WHITE STREAMLIT BUTTONS
           ========================================================== */
        button, .stButton > button, .st-key-button button {
            background: rgba(15,23,42,0.92) !important;
            color: #e2e8f0 !important;
            border-radius: 14px !important;
            border: 1px solid rgba(148,163,184,0.25) !important;
            padding: 0.6rem 1rem !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5) !important;
            transition: 0.15s ease-out !important;
        }

        button:hover, .stButton > button:hover {
            background: rgba(6,182,212,0.18) !important;
            border-color: rgba(34,197,94,0.9) !important;
            color: #f8fafc !important;
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.65) !important;
        }

        button:disabled, .stButton > button:disabled {
            background: rgba(20,27,45,0.6) !important;
            color: #94a3b8 !important;
            border: 1px solid rgba(75,85,99,0.4) !important;
            box-shadow: none !important;
            opacity: 0.4 !important;
        }

        /* -----------------------------------------------------------
           GLOBAL — FONT + BACKGROUND
        ----------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1000px circle at 6% -5%, rgba(34,197,94,0.22), transparent 45%),
                radial-gradient(900px circle at 100% 0%, rgba(6,182,212,0.20), transparent 45%),
                radial-gradient(1100px circle at 60% 110%, rgba(99,102,241,0.18), transparent 50%),
                linear-gradient(180deg, #030615 0%, #020617 55%, #020617 100%);
            color: #e6eaf2;
        }

        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1250px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        /* Header shell */
        .app-header-shell {
            background:
                linear-gradient(135deg,
                    rgba(34,197,94,0.18),
                    rgba(6,182,212,0.14),
                    rgba(2,6,23,0.98));
            border-radius: 22px;
            border: 1px solid rgba(148,163,184,0.35);
            padding: 12px 18px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.7);
            margin-bottom: 1.1rem;
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
                        #34d399,
                        #06b6d4 55%,
                        #6366f1);
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        }

        .badge-chip {
            padding: 6px 14px;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.35);
            background: rgba(2,6,23,0.7);
            display:flex;
            align-items:center;
            gap:7px;
            font-size:0.8rem;
            color:#e2e8f0;
        }

        .badge-dot {
            width:7px;
            height:7px;
            background:#22c55e;
            border-radius:999px;
            box-shadow:0 0 10px rgba(34,197,94,0.9);
        }

        /* Glass cards */
        .glass-card {
            background: linear-gradient(
                180deg,
                rgba(15,23,42,0.85) 0%,
                rgba(2,6,23,0.95) 100%
            );
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-radius: 20px;
            border: 1px solid rgba(148,163,184,0.18);
            padding: 18px 20px;
            box-shadow:
                0 10px 30px rgba(0,0,0,0.6),
                inset 0 1px 0 rgba(255,255,255,0.04);
        }

        .main-input-card { margin-bottom: 0.8rem; }
        .analysis-card   { margin-top: 0.3rem; }
        .side-card       { margin-bottom: 0.9rem; }

        .card-section-title {
            font-weight:700;
            font-size:0.9rem;
            margin-bottom:0.7rem;
            display:flex;
            gap:10px;
            align-items:center;
            color:#e2e8f0;
        }

        .card-section-title-icon {
            width:26px;
            height:26px;
            border-radius:999px;
            background:rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.12);
            display:flex;
            align-items:center;
            justify-content:center;
        }

        .card-section-body {
            font-size:0.82rem;
            color:#cbd5e1;
            line-height:1.45;
        }

        /* Text area */
        textarea {
            background: rgba(7,11,22,0.95) !important;
            border-radius: 14px !important;
            border: 1px solid rgba(148,163,184,0.30) !important;
            color: #e5e7eb !important;
            font-size:0.98rem !important;
        }
        textarea:focus {
            border-color: #06b6d4 !important;
            box-shadow: 0 0 0 1px #06b6d4 !important;
        }

        /* Analyze button */
        .analyze-button .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #22c55e 0%, #06b6d4 45%, #6366f1 100%) !important;
            border: none !important;
            padding: 0.75rem 1.1rem !important;
            border-radius: 999px !important;
            color: #030615 !important;
            font-weight: 800 !important;
            letter-spacing: 0.01em;
            box-shadow: 0 14px 40px rgba(34,197,94,0.30);
            transition: all 0.15s ease-out;
        }
        .analyze-button .stButton > button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 22px 55px rgba(0,0,0,0.7);
            filter:brightness(1.06);
        }
        .analyze-button .stButton > button:disabled {
            background: rgba(15,23,42,0.92) !important;
            color: #94a3b8 !important;
            border: 1px solid rgba(148,163,184,0.35) !important;
            box-shadow:none !important;
        }

        /* Example buttons */
        .example-btn .stButton > button,
        .example-btn button {
            width: 100% !important;
            text-align:left !important;
            white-space:normal!important;
            background: rgba(15,23,42,0.92) !important;
            border: 1px solid rgba(148,163,184,0.25) !important;
            border-radius: 14px !important;
            color: #e2e8f0 !important;
            padding: 0.65rem 0.95rem !important;
            font-size:0.84rem !important;
            box-shadow: 0 10px 28px rgba(0,0,0,0.55) !important;
            transition: transform 0.15s ease-out, box-shadow 0.15s ease-out, border-color 0.15s ease-out;
        }
        .example-btn .stButton > button:hover,
        .example-btn button:hover {
            background: rgba(6,182,212,0.18) !important;
            border-color: rgba(34,197,94,0.9) !important;
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(0,0,0,0.65) !important;
            color:#f8fafc !important;
        }

        /* Fix whiteness in uploader, selectbox, slider, dataframe */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(2,6,23,0.9) !important;
            border: 1px dashed rgba(148,163,184,0.45) !important;
            color: #e2e8f0 !important;
            border-radius: 14px !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: #e2e8f0 !important;
        }
        [data-baseweb="select"] > div {
            background: rgba(7,11,22,0.95) !important;
            border-color: rgba(148,163,184,0.35) !important;
            color: #e2e8f0 !important;
            border-radius: 12px !important;
        }
        [data-baseweb="select"] span {
            color: #e2e8f0 !important;
        }
        [data-testid="stSlider"] * {
            color: #e2e8f0 !important;
        }
        .stDataFrame, [data-testid="stDataFrame"] {
            background: rgba(2,6,23,0.95) !important;
        }

        /* Sentiment pill */
        .sentiment-pill {
            padding:6px 14px;
            border-radius:999px;
            font-size:0.86rem;
            font-weight:700;
            display:inline-flex;
            align-items:center;
            gap:6px;
        }
        .sentiment-positive {
            background:rgba(34,197,94,0.20);
            border:1px solid #22c55e;
            color:#bbf7d0;
        }
        .sentiment-negative {
            background:rgba(249,115,22,0.22);
            border:1px solid #f97316;
            color:#ffedd5;
        }
        .sentiment-neutral {
            background:rgba(156,163,175,0.20);
            border:1px solid #9ca3af;
            color:#e5e7eb;
        }

        /* Summary callout */
        .summary-callout {
            margin-top:0.7rem;
            padding:10px 12px;
            border-radius:14px;
            background: rgba(2,6,23,0.85);
            border:1px solid rgba(148,163,184,0.22);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
            font-size:0.9rem;
            color:#e2e8f0;
        }

        /* Stat pills row */
        .stat-row {
            display:flex;
            gap:10px;
            margin-top:0.9rem;
            flex-wrap:wrap;
        }
        .stat-pill {
            flex:1;
            min-width:140px;
            padding:10px 12px;
            border-radius:14px;
            background:rgba(2,6,23,0.95);
            border:1px solid rgba(148,163,184,0.25);
            font-size:0.82rem;
        }
        .stat-pill .label {
            color:#9ca3af; margin-bottom:3px;
        }
        .stat-pill .value {
            font-size:1.02rem; font-weight:700; color:#e5e7eb;
        }

        /* Section subtitles */
        .section-subtitle {
            margin-top:1.1rem;
            font-size:0.92rem;
            font-weight:800;
            color:#e2e8f0;
        }

        /* Tiny divider */
        .soft-divider {
            height:1px;
            background: linear-gradient(90deg, transparent, rgba(148,163,184,0.35), transparent);
            margin: 0.9rem 0 0.5rem 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# SHAP + XAI HELPERS
# -------------------------------------------------------------------

def value_to_color(v, max_abs):
    if max_abs <= 0:
        return "transparent"
    v = max(-max_abs, min(max_abs, v))
    alpha = 0.20 + 0.80 * (abs(v) / max_abs)
    if v > 0:
        return f"rgba(34,197,94,{alpha})"
    else:
        return f"rgba(249,115,22,{alpha})"


def render_inline_heatmap(tokens, values):
    if not tokens:
        return ""
    max_abs = max(abs(v) for v in values) or 1
    html_parts = []
    for tok, val in zip(tokens, values):
        bg = value_to_color(val, max_abs)
        safe_tok = tok.replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(
            f"<span title='SHAP: {val:.4f}' "
            f"style='background:{bg}; padding:3px 7px; border-radius:9px; margin-right:4px; "
            f"line-height:2.2; display:inline-block; border:1px solid rgba(255,255,255,0.12);'>"
            f"{safe_tok}</span>"
        )

    max_pos = max([v for v in values if v > 0], default=0)
    max_neg = min([v for v in values if v < 0], default=0)
    legend = (
        f"<div style='margin-top:8px; font-size:0.78rem; color:#9ca3af;'>"
        f"Strongest positive: <b style='color:#bbf7d0'>{max_pos:.3f}</b> · "
        f"Strongest negative: <b style='color:#ffedd5'>{max_neg:.3f}</b>"
        f"</div>"
    )

    return " ".join(html_parts) + legend


def merge_subword_tokens(tokens, values):
    merged_tokens = []
    merged_values = []
    current_tok = ""
    current_val = 0.0

    for tok, val in zip(tokens, values):
        if tok.startswith("##"):
            current_tok += tok[2:]
            current_val += val
        else:
            if current_tok:
                merged_tokens.append(current_tok)
                merged_values.append(current_val)
            current_tok = tok
            current_val = val

    if current_tok:
        merged_tokens.append(current_tok)
        merged_values.append(current_val)

    return merged_tokens, merged_values


def shap_summary(tokens, values, top_k=3):
    pairs = sorted(zip(tokens, values), key=lambda x: abs(x[1]), reverse=True)
    pos = [t for t, v in pairs if v > 0][:top_k]
    neg = [t for t, v in pairs if v < 0][:top_k]

    parts = []
    if pos:
        parts.append("positive drivers: " + ", ".join([f"<b>{p}</b>" for p in pos]))
    if neg:
        parts.append("negative drivers: " + ", ".join([f"<b>{n}</b>" for n in neg]))

    return " | ".join(parts) if parts else "No strong drivers found."


def plot_sentiment_gauge(p_pos):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(p_pos) * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 0},
                "bar": {"color": "rgba(0,0,0,0)"},
                "steps": [
                    {"range": [0, 40], "color": "rgba(249,115,22,0.5)"},
                    {"range": [40, 60], "color": "rgba(156,163,175,0.5)"},
                    {"range": [60, 100], "color": "rgba(34,197,94,0.5)"},
                ],
                "threshold": {
                    "line": {"color": "#e2e8f0", "width": 3},
                    "thickness": 0.75,
                    "value": float(p_pos) * 100,
                },
            },
            title={"text": "Positive probability", "font": {"size": 12, "color": "#9ca3af"}},
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=12, r=12, t=25, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_shap_bars(tokens, shap_values):
    df = pd.DataFrame({"token": tokens, "shap_value": shap_values})
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).head(12)
    df = df.iloc[::-1]

    colors = ["#22c55e" if v > 0 else "#f97316" for v in df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=df["shap_value"],
            y=df["token"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>SHAP impact: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Impact on positive sentiment",
            gridcolor="rgba(148,163,184,0.12)",
            zerolinecolor="rgba(148,163,184,0.35)",
            tickfont=dict(color="#cbd5e1"),
            #titlefont=dict(color="#cbd5e1"),
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=13)),
        showlegend=False,
    )

    fig.add_vline(x=0, line_width=1.2, line_color="rgba(226,232,240,0.45)")
    return fig


@st.cache_data(show_spinner=False)
def batch_predict(texts):
    rows = []
    for t in texts:
        score, p_pos = positivity_score(t)
        label, _ = predict_label(t)
        rows.append({"review": t, "score": score, "p_pos": float(p_pos), "label": label})
    return pd.DataFrame(rows)


def guess_text_column(df):
    candidates = ["review", "text", "sentence", "message", "comment", "content"]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    return obj_cols[0] if obj_cols else df.columns[0]


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

    # HEADER
    st.markdown(
        """
        <div class="app-header-shell">
          <div class="app-header">
            <div class="app-header-left">
              <div class="logo-pill">☻</div>
              <div>
                <div class="app-title-main" style="font-weight:800; font-size:1.15rem;">
                  Explainable Sentiment Analyzer
                </div>
                <div class="app-title-sub" style="font-size:0.84rem; color:#cbd5e1;">
                  Understand sentiment instantly and see which words drive it
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

    col_main, col_side = st.columns([2.2, 0.8], gap="large")

    # RIGHT PANEL
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
                • Sentence-level driver highlights<br>
                • Batch CSV sentiment mode
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # LEFT PANEL
    with col_main:

        # ======================================================
        # BATCH CSV ANALYSIS CARD
        # ======================================================
        st.markdown('<div class="glass-card main-input-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-section-title">
                <div class="card-section-title-icon">📂</div>
                <span>Batch analysis (CSV upload)</span>
            </div>
            <div class="card-section-body">
                <div style="margin-bottom:6px;">
                    <b>CSV format:</b> one row per review, with a text column like <code>review</code>.
                    Example columns: <code>review_id, review, product, rating</code>.
                </div>
                If your file contains a non-data first row (like a note), enable “Skip first data row”.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Upload CSV containing reviews", type=["csv"])

        if uploaded is not None:
            try:
                df_csv = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                df_csv = None

            if df_csv is not None and len(df_csv.columns) > 0:
                default_col = guess_text_column(df_csv)
                review_col = st.selectbox(
                    "Select the review text column",
                    df_csv.columns,
                    index=list(df_csv.columns).index(default_col),
                    help="Pick the column that contains the actual review text."
                )

                skip_first = st.checkbox(
                    "Skip first data row (recommended if your CSV has a note row)",
                    value=False
                )

                max_rows = st.slider(
                    "Max rows to score (safety for large files)",
                    min_value=50,
                    max_value=min(5000, max(50, len(df_csv))),
                    value=min(500, len(df_csv)),
                    step=50,
                )

                run_batch = st.button("Analyze CSV")

                if run_batch:
                    series = df_csv[review_col].dropna().astype(str)
                    if skip_first and len(series) > 0:
                        series = series.iloc[1:]
                    texts = series.tolist()[:max_rows]

                    if len(texts) == 0:
                        st.error("No reviews found in that column after filtering.")
                    elif all(t.strip().isdigit() for t in texts[:10]):
                        st.error(
                            "It looks like you selected a numeric ID column. "
                            "Please select a text column (e.g., 'review')."
                        )
                    else:
                        with st.spinner("Running sentiment analysis on reviews..."):
                            batch_results = batch_predict(texts)

                        st.success(f"Scored {len(batch_results)} reviews.")

                        avg_score = float(batch_results["score"].mean())
                        pct_pos = float((batch_results["label"] == "positive").mean() * 100)
                        pct_neg = 100 - pct_pos

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Average positivity", f"{avg_score:.1f}/100")
                        m2.metric("Percent positive", f"{pct_pos:.1f}%")
                        m3.metric("Percent negative", f"{pct_neg:.1f}%")

                        st.dataframe(batch_results, use_container_width=True, height=320)

                        csv_bytes = batch_results.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results CSV",
                            data=csv_bytes,
                            file_name="sentiment_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ======================================================
        # SINGLE TEXT INPUT CARD
        # ======================================================
        st.markdown('<div class="glass-card main-input-card">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="card-section-title">
                <div class="card-section-title-icon">📄</div>
                <span>Single review analysis</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        text = st.text_area(
            "",
            key="text_input",
            height=175,
            placeholder="Type or paste a sentence..."
        )
        st.markdown('<div class="analyze-button">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Sentiment", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:0.75rem; color:#9ca3af; margin-top:6px;'>Press Analyze Sentiment to run the model.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if analyze_clicked and text.strip():
            progress = st.progress(0)
            for p in [15, 35, 60]:
                progress.progress(p)

            score, p_pos = positivity_score(text)
            label, _ = predict_label(text)
            confidence = max(p_pos, 1 - p_pos)
            conf_pct = int(confidence * 100)

            if score > 60:
                sentiment = "Positive"
                pill = "sentiment-positive"
                emoji = "😄"
            elif score < 40:
                sentiment = "Negative"
                pill = "sentiment-negative"
                emoji = "😕"
            else:
                sentiment = "Neutral"
                pill = "sentiment-neutral"
                emoji = "😐"

            with st.spinner("Computing SHAP explanations..."):
                tokens, shap_values, base = get_token_importances(text)

            progress.progress(100)
            progress.empty()

            tokens, shap_values = merge_subword_tokens(tokens, shap_values)

            pos_count = sum(v > 0.01 for v in shap_values)
            neg_count = sum(v < -0.01 for v in shap_values)
            neut_count = len(tokens) - pos_count - neg_count

            pos_total = sum(v for v in shap_values if v > 0)
            neg_total = -sum(v for v in shap_values if v < 0)
            total = pos_total + neg_total + 1e-9
            pos_pct = int(100 * pos_total / total)
            neg_pct = 100 - pos_pct

            why = shap_summary(tokens, shap_values)

            st.markdown('<div class="glass-card analysis-card">', unsafe_allow_html=True)

            left_top, right_top = st.columns([1.2, 1.0], gap="large")
            with left_top:
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="sentiment-pill {pill}">{emoji} {sentiment}</div>
                        <div style="font-size:1.5rem; font-weight:800;">{conf_pct}%</div>
                    </div>
                    <div class="summary-callout">
                        <div style="font-weight:800; margin-bottom:2px;">Key drivers</div>
                        {why}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right_top:
                fig_gauge = plot_sentiment_gauge(p_pos)
                st.plotly_chart(
                    fig_gauge,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div class="stat-row">
                    <div class="stat-pill">
                        <div class="label">✅ Positive tokens</div>
                        <div class="value">{pos_count}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="label">😬 Negative tokens</div>
                        <div class="value">{neg_count}</div>
                    </div>
                    <div class="stat-pill">
                        <div class="label">😐 Neutral tokens</div>
                        <div class="value">{neut_count}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="margin-top:0.9rem;">
                  <div style="font-size:0.8rem; color:#9ca3af;">Contribution balance</div>
                  <div style="height:12px; border-radius:999px; overflow:hidden; display:flex;
                              border:1px solid rgba(75,85,99,0.8); background:rgba(15,23,42,0.8);">
                    <div style="width:{pos_pct}%; background:#22c55e;"></div>
                    <div style="width:{neg_pct}%; background:#f97316;"></div>
                  </div>
                  <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#9ca3af; margin-top:4px;">
                    <span>Positive {pos_pct}%</span>
                    <span>Negative {neg_pct}%</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div style="margin-top:0.9rem;">
                    <div style="font-size:0.8rem; color:#9ca3af;">Model confidence</div>
                    <div style="height:10px; border-radius:999px; background:rgba(15,23,42,0.8);
                                border:1px solid rgba(75,85,99,0.8);">
                        <div style="height:10px; width:{conf_pct}%;
                                    background:linear-gradient(90deg,#22c55e,#06b6d4,#6366f1);
                                    border-radius:999px;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

            tab_explain, tab_chart = st.tabs(["Explanation", "Top Drivers"])

            with tab_explain:
                st.markdown(
                    "<div class='section-subtitle'>Sentence-level explanation (SHAP heatmap)</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(render_inline_heatmap(tokens, shap_values), unsafe_allow_html=True)

                st.markdown(
                    """
                    <div style="margin-top:10px; font-size:0.75rem;">
                        <span style="background:rgba(34,197,94,0.22); border:1px solid #22c55e; padding:3px 8px; border-radius:999px; margin-right:8px; color:#e2e8f0;">
                            Positive contribution
                        </span>
                        <span style="background:rgba(249,115,22,0.22); border:1px solid #f97316; padding:3px 8px; border-radius:999px; color:#e2e8f0;">
                            Negative contribution
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with tab_chart:
                st.markdown(
                    "<div class='section-subtitle'>Most influential words</div>",
                    unsafe_allow_html=True,
                )
                fig_bar = plot_shap_bars(tokens, shap_values)
                st.plotly_chart(
                    fig_bar,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            st.markdown("</div>", unsafe_allow_html=True)

    # FOOTER
    st.markdown(
        """
        <div style="margin-top:1.8rem; color:#6b7280; font-size:0.8rem;">
            Built by <b>Eric Ortega Rodriguez</b> &amp; <b>Diya Mirji</b> · For AIPI 590: Emerging Trends in Explainable AI
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

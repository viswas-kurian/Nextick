import streamlit as st
import pandas as pd
import json
import os
import logging
from src.predict import fetch_and_predict
from src.features import create_features_and_target

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nextick — Stock Movement Analysis",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# DARK THEME CSS
# ==========================================================
st.markdown("""
<style>
/* ── Import Inter font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global resets ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #0E1117 !important;
    color: #E5E7EB;
}

h1, h2, h3, h4, h5, h6 {
    color: #F1F5F9 !important;
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1140px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #111318;
    border-right: 1px solid #1E2028;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] .stCaption {
    color: #9CA3AF;
}

/* ── Header ── */
.ntk-header {
    padding: 0.2rem 0 0.6rem 0;
    border-bottom: 1px solid #262830;
    margin-bottom: 1.6rem;
}
.ntk-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #F1F5F9;
    margin: 0;
    line-height: 1.2;
}
.ntk-subtitle {
    font-size: 0.82rem;
    font-weight: 400;
    color: #6B7280;
    margin-top: 0.15rem;
    line-height: 1.4;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.6rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #262830;
}

/* ── Prediction display ── */
.pred-card {
    background: #16181D;
    border: 1px solid #262830;
    border-radius: 4px;
    padding: 1.4rem 1.2rem;
    text-align: center;
}
.pred-direction {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.pred-horizon-label {
    font-size: 0.7rem;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #6B7280;
    margin-bottom: 0.25rem;
}
.pred-probs {
    margin-top: 0.8rem;
    font-size: 0.82rem;
    color: #9CA3AF;
    line-height: 1.8;
}
.pred-probs .val {
    font-weight: 600;
    color: #D1D5DB;
}

/* ── Probability bar ── */
.prob-bar-track {
    width: 100%;
    height: 6px;
    background: #374151;
    border-radius: 3px;
    margin: 0.6rem 0 0.3rem 0;
    overflow: hidden;
    display: flex;
}
.prob-bar-fill-up {
    height: 100%;
    background: #22C55E;
    border-radius: 3px 0 0 3px;
}
.prob-bar-fill-down {
    height: 100%;
    background: #EF4444;
    border-radius: 0 3px 3px 0;
}

/* ── Metric tile ── */
.metric-tile {
    background: #16181D;
    border: 1px solid #262830;
    border-radius: 4px;
    padding: 0.9rem 1rem;
}
.metric-tile .label {
    font-size: 0.68rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B7280;
    margin-bottom: 0.25rem;
}
.metric-tile .value {
    font-size: 1.15rem;
    font-weight: 600;
    color: #E5E7EB;
    line-height: 1.2;
}
.metric-tile .delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.1rem;
}

/* ── Data table styling ── */
.indicator-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.indicator-table th {
    text-align: left;
    font-weight: 600;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6B7280;
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid #262830;
}
.indicator-table td {
    padding: 0.45rem 0.6rem;
    border-bottom: 1px solid #1E2028;
    color: #D1D5DB;
}
.indicator-table tr:last-child td {
    border-bottom: none;
}
.indicator-table .val-cell {
    font-variant-numeric: tabular-nums;
    text-align: right;
    font-weight: 500;
}

/* ── Callout box ── */
.callout {
    background: #16181D;
    border-left: 3px solid #374151;
    padding: 0.7rem 0.9rem;
    font-size: 0.78rem;
    color: #9CA3AF;
    line-height: 1.55;
    margin-top: 0.7rem;
    border-radius: 0 3px 3px 0;
}

/* ── Footer ── */
.ntk-footer {
    font-size: 0.72rem;
    color: #4B5563;
    border-top: 1px solid #1E2028;
    padding-top: 0.9rem;
    margin-top: 2.5rem;
    line-height: 1.4;
}

/* ── Tab styling tweaks ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #262830;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.6rem 1.1rem;
    color: #6B7280;
}
.stTabs [aria-selected="true"] {
    color: #E5E7EB;
    border-bottom-color: #E5E7EB;
}

/* ── Streamlit metric override ── */
[data-testid="stMetricValue"] {
    font-size: 1rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-size: 0.82rem;
    font-weight: 500;
}

/* ── Hide Streamlit default branding ── */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD METADATA (cached)
# ==========================================================
@st.cache_data(show_spinner=False)
def load_metadata():
    metadata_path = os.path.join(os.path.dirname(__file__), 'models/metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}

metadata = load_metadata()

# ==========================================================
# CACHED PREDICTION WRAPPER
# ==========================================================
@st.cache_data(ttl=900, show_spinner=False)
def get_stock_prediction(ticker: str):
    return fetch_and_predict(ticker)

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div class="ntk-header">
    <div class="ntk-title">NEXTICK</div>
    <div class="ntk-subtitle">Machine Learning-Based Stock Movement Analysis</div>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# STOCK UNIVERSE (unchanged from backend)
# ==========================================================
TICKER_UNIVERSE = {
    "Technology": {
        "Apple Inc. (AAPL)": "AAPL",
        "Microsoft Corp. (MSFT)": "MSFT",
        "Alphabet Inc. (GOOGL)": "GOOGL",
        "Amazon.com Inc. (AMZN)": "AMZN",
        "Meta Platforms Inc. (META)": "META",
        "NVIDIA Corp. (NVDA)": "NVDA",
        "Broadcom Inc. (AVGO)": "AVGO",
        "Oracle Corp. (ORCL)": "ORCL",
        "Adobe Inc. (ADBE)": "ADBE",
        "Cisco Systems Inc. (CSCO)": "CSCO",
        "Intel Corp. (INTC)": "INTC",
    },
    "Financials": {
        "JPMorgan Chase & Co. (JPM)": "JPM",
        "Visa Inc. (V)": "V",
        "Mastercard Inc. (MA)": "MA",
        "Bank of America Corp. (BAC)": "BAC",
        "Wells Fargo & Co. (WFC)": "WFC",
        "Goldman Sachs Group (GS)": "GS",
    },
    "Healthcare": {
        "Johnson & Johnson (JNJ)": "JNJ",
        "UnitedHealth Group Inc. (UNH)": "UNH",
        "Pfizer Inc. (PFE)": "PFE",
        "AbbVie Inc. (ABBV)": "ABBV",
        "Merck & Co. Inc. (MRK)": "MRK",
        "Eli Lilly and Co. (LLY)": "LLY",
    },
    "Consumer Goods & Services": {
        "Walmart Inc. (WMT)": "WMT",
        "Costco Wholesale Corp. (COST)": "COST",
        "The Coca-Cola Co. (KO)": "KO",
        "PepsiCo Inc. (PEP)": "PEP",
        "McDonald's Corp. (MCD)": "MCD",
        "Procter & Gamble Co. (PG)": "PG",
        "Nike Inc. (NKE)": "NKE",
    },
    "Industrial & Energy": {
        "Caterpillar Inc. (CAT)": "CAT",
        "General Electric Co. (GE)": "GE",
        "Exxon Mobil Corp. (XOM)": "XOM",
        "Chevron Corp. (CVX)": "CVX",
        "The Boeing Co. (BA)": "BA",
        "Honeywell International Inc. (HON)": "HON",
    }
}

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown("#### Stock Selection")
selected_sector = st.sidebar.selectbox("Sector", list(TICKER_UNIVERSE.keys()), label_visibility="visible")
sector_stocks = TICKER_UNIVERSE[selected_sector]
selected_stock_name = st.sidebar.selectbox("Company", list(sector_stocks.keys()), label_visibility="visible")
selected_ticker = sector_stocks[selected_stock_name]

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"**Ticker:** `{selected_ticker}`  \n"
    f"**Horizon:** Next Trading Day  \n"
    f"**Model:** {metadata.get('selected_model', 'Random Forest')}  \n"
    f"**Universe:** {metadata.get('num_stocks', 36)} stocks"
)

st.sidebar.markdown("---")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Data: Yahoo Finance API  \nLive OHLCV data retrieved at query time.")

# ==========================================================
# MAIN TABS
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Market Analysis", "Model & Evaluation", "About Nextick"])

# ==========================================================
# TAB 1: MARKET ANALYSIS
# ==========================================================
with tab1:
    with st.spinner(f"Loading {selected_ticker}..."):
        try:
            data = get_stock_prediction(selected_ticker)

            # ── Company name + date ──
            st.markdown(f"### {selected_stock_name}")
            st.caption(f"As of {data['latest_date']}  ·  {selected_ticker}  ·  {selected_sector}")

            # ── Market Snapshot ──
            st.markdown('<div class="section-label">Market Snapshot</div>', unsafe_allow_html=True)

            change_sign = "+" if data['daily_change'] >= 0 else ""
            change_color = "#22C55E" if data['daily_change'] >= 0 else "#EF4444"

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(f"""<div class="metric-tile">
                    <div class="label">Latest Close</div>
                    <div class="value">${data['latest_close']:.2f}</div>
                    <div class="delta" style="color:{change_color}">{change_sign}{data['daily_change']:.2f} ({change_sign}{data['daily_change_pct']:.2f}%)</div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""<div class="metric-tile">
                    <div class="label">Previous Close</div>
                    <div class="value">${data['previous_close']:.2f}</div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""<div class="metric-tile">
                    <div class="label">Trading Volume</div>
                    <div class="value">{data['volume']:,}</div>
                </div>""", unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""<div class="metric-tile">
                    <div class="label">Daily Return</div>
                    <div class="value" style="color:{change_color}">{change_sign}{data['daily_change_pct']:.2f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")  # spacer

            # ── Prediction + Chart row ──
            pred_col, chart_col = st.columns([1, 2])

            with pred_col:
                st.markdown('<div class="section-label">Next-Day Model Estimate</div>', unsafe_allow_html=True)

                pred_label = data["prediction"]
                pred_color = "#22C55E" if pred_label == "UP" else "#EF4444"
                pred_arrow = "▲" if pred_label == "UP" else "▼"
                prob_up_pct = data['probability_up'] * 100
                prob_down_pct = data['probability_down'] * 100

                st.markdown(f"""
                <div class="pred-card">
                    <div class="pred-horizon-label">Predicted Direction</div>
                    <div class="pred-direction" style="color: {pred_color};">
                        {pred_arrow} {pred_label}
                    </div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill-up" style="width: {prob_up_pct:.1f}%"></div>
                        <div class="prob-bar-fill-down" style="width: {prob_down_pct:.1f}%"></div>
                    </div>
                    <div class="pred-probs">
                        UP: <span class="val">{prob_up_pct:.1f}%</span> &nbsp;&middot;&nbsp;
                        DOWN: <span class="val">{prob_down_pct:.1f}%</span>
                    </div>
                    <div style="margin-top:0.6rem; font-size:0.68rem; color:#4B5563;">
                        Prediction horizon: Next trading day
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""<div class="callout">
                    Model probabilities represent the distribution learned from historical data.
                    They are not guarantees of future price movement or financial advice.
                </div>""", unsafe_allow_html=True)

            with chart_col:
                st.markdown('<div class="section-label">Closing Price — Last 6 Months</div>', unsafe_allow_html=True)
                hist_df = data["historical_data"].copy()
                hist_df['Date'] = pd.to_datetime(hist_df['Date'])
                hist_df = hist_df.sort_values('Date').set_index('Date')
                st.line_chart(hist_df['Close'], height=260, use_container_width=True)

            # ── Technical Indicators ──
            with st.expander("Technical Indicators (Inference Input)"):
                st.caption("Computed strictly from data available at prediction time. These are the features fed to the trained model.")

                raw_hist = data["historical_data"].copy()
                raw_hist['Ticker'] = selected_ticker
                feats = create_features_and_target(raw_hist, is_training=False).iloc[-1:]

                # Build organized indicator table
                indicators = [
                    ("Moving Averages", [
                        ("SMA 5", f"${feats['SMA_5'].values[0]:.2f}"),
                        ("SMA 10", f"${feats['SMA_10'].values[0]:.2f}"),
                        ("SMA 20", f"${feats['SMA_20'].values[0]:.2f}"),
                        ("EMA 10", f"${feats['EMA_10'].values[0]:.2f}"),
                        ("EMA 20", f"${feats['EMA_20'].values[0]:.2f}"),
                    ]),
                    ("Returns & Momentum", [
                        ("Daily Return", f"{feats['Daily_Return'].values[0]:+.2%}"),
                        ("1-Day Lag Return", f"{feats['Lag1_Return'].values[0]:+.2%}"),
                        ("2-Day Lag Return", f"{feats['Lag2_Return'].values[0]:+.2%}"),
                        ("5-Day Return", f"{feats['Return_5d'].values[0]:+.2%}"),
                        ("Intraday (Open→Close)", f"{feats['Open_Close_Range'].values[0]:+.2%}"),
                    ]),
                    ("Volatility & Volume", [
                        ("10-Day Volatility", f"{feats['Rolling_Vol_10'].values[0]:.4f}"),
                        ("20-Day Volatility", f"{feats['Rolling_Vol_20'].values[0]:.4f}"),
                        ("Intraday Range (H-L)", f"{feats['High_Low_Range'].values[0]:.2%}"),
                        ("Volume Change", f"{feats['Volume_Change'].values[0]:+.2%}"),
                        ("Volume vs 5D Avg", f"{feats['Volume_to_MA5'].values[0]:+.2%}"),
                    ]),
                    ("Relative Position", [
                        ("Close / SMA 5", f"{feats['Close_to_SMA5'].values[0]:+.2%}"),
                        ("Close / SMA 10", f"{feats['Close_to_SMA10'].values[0]:+.2%}"),
                        ("Close / SMA 20", f"{feats['Close_to_SMA20'].values[0]:+.2%}"),
                        ("SMA 5 / SMA 20", f"{feats['SMA5_to_SMA20'].values[0]:+.2%}"),
                        ("EMA 10 / EMA 20", f"{feats['EMA10_to_EMA20'].values[0]:+.2%}"),
                    ]),
                ]

                icol1, icol2 = st.columns(2)
                for idx, (group_name, items) in enumerate(indicators):
                    col = icol1 if idx % 2 == 0 else icol2
                    with col:
                        rows_html = ""
                        for name, val in items:
                            rows_html += f'<tr><td>{name}</td><td class="val-cell">{val}</td></tr>\n'
                        st.markdown(f"""
                        <table class="indicator-table">
                            <thead><tr><th colspan="2">{group_name}</th></tr></thead>
                            <tbody>{rows_html}</tbody>
                        </table><br>
                        """, unsafe_allow_html=True)

        except Exception as e:
            logging.getLogger(__name__).error(f"Prediction failed for {selected_ticker}: {e}", exc_info=True)
            st.error(f"Market data could not be retrieved for {selected_stock_name}. Please try again.")
            with st.expander("Debug information"):
                st.code(f"{type(e).__name__}: {str(e)}")


# ==========================================================
# TAB 2: MODEL & EVALUATION
# ==========================================================
with tab2:
    st.markdown("### Model Evaluation")
    st.caption("All metrics computed on a held-out test set using a strict chronological split. No future data was used during training or evaluation.")

    # ── Key facts ──
    st.markdown('<div class="section-label">Training Configuration</div>', unsafe_allow_html=True)

    kf1, kf2, kf3, kf4, kf5 = st.columns(5)
    kf1.metric("Model", metadata.get("selected_model", "Random Forest"))
    kf2.metric("Training Universe", f"{metadata.get('num_stocks', 36)} Stocks")
    kf3.metric("Training Period", "2015–2024")
    kf4.metric("Input Features", f"{len(metadata.get('feature_columns', []))}")
    kf5.metric("Test ROC-AUC", f"{metadata.get('metrics', {}).get('Random Forest', {}).get('ROC-AUC', 0.5012):.4f}")

    st.markdown("")

    # ── Dataset split ──
    st.markdown('<div class="section-label">Chronological Train / Test Split</div>', unsafe_allow_html=True)

    ds1, ds2, ds3 = st.columns(3)
    ds1.metric("Total Processed Rows", f"{metadata.get('total_processed_rows', 105444):,}")
    ds2.metric("Training Rows (80%)", f"{metadata.get('train_rows', 84348):,}")
    ds3.metric("Test Rows (20%)", f"{metadata.get('test_rows', 21096):,}")

    st.markdown(
        f"**Training:** {metadata.get('train_date_range', ['2015-02-02'])[0]} → "
        f"{metadata.get('train_date_range', ['', '2024-05-22'])[1]}  \n"
        f"**Test (out-of-sample):** {metadata.get('test_date_range', ['2024-05-23'])[0]} → "
        f"{metadata.get('test_date_range', ['', '2026-09-24'])[1]}  \n"
        "Split method: Global chronological date cutoff across all stocks. Zero lookahead leakage."
    )

    st.markdown("---")

    # ── Benchmark table ──
    st.markdown('<div class="section-label">Model Benchmark — Test Set Performance</div>', unsafe_allow_html=True)

    if "metrics" in metadata:
        metrics_dict = metadata["metrics"]
        rows = []
        for m_name, m_vals in metrics_dict.items():
            rows.append({
                "Model": m_name,
                "Accuracy": f"{m_vals['Accuracy']:.4f}",
                "Precision": f"{m_vals['Precision']:.4f}",
                "Recall": f"{m_vals['Recall']:.4f}",
                "F1-Score": f"{m_vals['F1-score']:.4f}",
                "ROC-AUC": f"{m_vals['ROC-AUC']:.4f}"
            })
        m_df = pd.DataFrame(rows)
        st.dataframe(m_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="callout">
        The Random Forest achieves an ROC-AUC of 0.5012 on the out-of-sample test set —
        marginally above the 0.500 random baseline. This is consistent with financial literature:
        daily price direction in large-cap equity markets approximates a random walk when
        analyzed solely through historical technical indicators.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Plots ──
    st.markdown('<div class="section-label">Evaluation Charts</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**ROC Curves**")
        roc_path = os.path.join(os.path.dirname(__file__), 'plots/roc_curves.png')
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Available after running the training pipeline.")
    with p2:
        st.markdown("**Feature Importance (Random Forest)**")
        fi_path = os.path.join(os.path.dirname(__file__), 'plots/feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Available after running the training pipeline.")


# ==========================================================
# TAB 3: ABOUT NEXTICK
# ==========================================================
with tab3:
    st.markdown("### About Nextick")

    st.markdown("""
    Nextick uses historical market data and machine learning to estimate whether a stock's
    next trading day closing price will be higher or lower than the current close.

    The system downloads daily OHLCV data from Yahoo Finance, computes 26 technical features
    (moving averages, momentum indicators, volatility measures, and normalized ratios), and
    feeds them into a trained Random Forest classifier.
    """)

    st.markdown("---")

    st.markdown("#### How It Works")
    st.markdown("""
    ```
    1. DATA — Download 10 years of daily OHLCV data for 36 large-cap US stocks

    2. FEATURES — Compute 26 technical indicators per stock per day
       Momentum:   Daily Return, Lag Returns, 5-Day Return
       Trend:      SMA (5, 10, 20), EMA (10, 20)
       Volatility: 10-day and 20-day rolling standard deviation
       Volume:     Volume change, 5-day volume mean, volume ratio
       Normalized: Close/SMA ratios, High-Low range, Intraday return

    3. TARGET — Binary classification
       UP (1)   = Next day's Close > Today's Close
       DOWN (0) = Next day's Close ≤ Today's Close

    4. SPLIT — Chronological 80/20 date cutoff (no future leakage)
       Training: 2015–2024  |  Test: 2024–2026

    5. MODELS — Baseline, Logistic Regression, Decision Tree, Random Forest
       Selected: Random Forest (highest test ROC-AUC)

    6. INFERENCE — Live OHLCV fetch → same feature pipeline → model prediction
    ```
    """)

    st.markdown("---")

    st.markdown("#### Limitations & Disclaimer")
    st.markdown("""
    - **Not financial advice.** This application does not provide trading signals, buy/sell
      recommendations, or investment guidance of any kind.
    - **Near-random performance.** The model achieves a test ROC-AUC of 0.5012 — essentially
      random. Daily stock movements in liquid markets are influenced by macroeconomic events,
      earnings, policy decisions, and order flow that cannot be captured by historical
      technical indicators alone.
    - **Survivorship bias.** The stock universe consists of currently established large-cap
      companies.
    - **Model probabilities** represent learned classifier distributions from historical
      training data. They are not guarantees of future price movement.
    """)

    st.markdown("---")

    st.markdown("#### Technical Stack")
    st.markdown("""
    Python · scikit-learn · pandas · yfinance · Streamlit

    **Data:** Yahoo Finance API (daily OHLCV)
    **Model:** Random Forest (150 estimators, max depth 6, min samples leaf 50)
    **Features:** 26 technical indicators
    **Evaluation:** Chronological train/test split across 36 stocks
    """)


# ==========================================================
# FOOTER
# ==========================================================
st.markdown("""
<div class="ntk-footer">
    NEXTICK &middot; Historical market data and machine learning for next-day directional analysis.
    &middot; Data from Yahoo Finance. Not financial advice.
</div>
""", unsafe_allow_html=True)

"""
streamlit_app.py
-----------------
Interactive TradeSense AI dashboard.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.backtest import run_backtest
from src.data_loader import load_price_history
from src.model import build_dataset, predict_latest, prepare_features, train_model

st.set_page_config(page_title="TradeSense AI", page_icon="\U0001F4C8", layout="wide")

st.title("TradeSense AI")
st.caption("AI-powered technical-analysis trading signals — Buy / Sell / Hold")

with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Ticker symbol", value="AAPL").upper().strip()
    period = st.selectbox("History window", ["1y", "2y", "3y", "5y"], index=2)
    horizon = st.slider("Signal horizon (days ahead)", 3, 15, 5)
    threshold = st.slider("BUY/SELL return threshold", 0.005, 0.05, 0.015, step=0.005)
    offline = st.checkbox("Force offline demo data", value=False)
    run_button = st.button("Run TradeSense AI", type="primary")

if run_button:
    with st.spinner(f"Loading data for {symbol} ..."):
        raw_df, source = load_price_history(symbol, period=period, force_offline=offline)
        st.info(f"Data source: **{source}**" + (" (offline synthetic demo data)" if source == "synthetic" else ""))

    with st.spinner("Engineering features and training model ..."):
        dataset = build_dataset(raw_df, horizon=horizon, threshold=threshold)
        model, metrics = train_model(dataset)
        live_features = prepare_features(raw_df)
        latest_signal = predict_latest(model, live_features)
        bt = run_backtest(model, dataset)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Signal", latest_signal["signal"])
    col2.metric("Confidence", f"{latest_signal['confidence']*100:.1f}%")
    col3.metric("Strategy Return", f"{bt['total_return_pct']}%")
    col4.metric("Buy & Hold Return", f"{bt['buy_and_hold_return_pct']}%")

    st.subheader(f"{symbol} Price Chart")
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=raw_df.index,
                open=raw_df["Open"],
                high=raw_df["High"],
                low=raw_df["Low"],
                close=raw_df["Close"],
                name=symbol,
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strategy vs. Buy & Hold Equity Curve")
    equity_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(bt["dates"]),
            "TradeSense AI Strategy": bt["equity_curve"],
            "Buy & Hold": bt["buy_hold_curve"],
        }
    ).set_index("Date")
    st.line_chart(equity_df)

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("Model Performance")
        st.metric("Cross-Validated Accuracy", f"{metrics['cv_mean_accuracy']*100:.1f}%")
        st.metric("Sharpe Ratio", bt["sharpe_ratio"])
        st.metric("Max Drawdown", f"{bt['max_drawdown_pct']}%")
        st.metric("Number of Trades", bt["num_trades"])

    with col6:
        st.subheader("Top Predictive Features")
        importances = pd.DataFrame(
            list(metrics["feature_importances"].items())[:8],
            columns=["Feature", "Importance"],
        ).set_index("Feature")
        st.bar_chart(importances)

    st.subheader("Signal Probability Breakdown")
    st.json(latest_signal["probabilities"])

else:
    st.write("Set a ticker in the sidebar and click **Run TradeSense AI** to generate a live signal.")
    st.write(
        "No internet in this environment? Check **Force offline demo data** to run the full "
        "pipeline on realistic synthetic price data."
    )

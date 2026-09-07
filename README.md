# TradeSense AI

**AI-powered technical analysis and trading-signal generation for stock markets.**

Alibaba Cloud AI Hackathon Pakistan 2026 — Submission

## Problem

Retail traders and early-stage investors in Pakistan and elsewhere rarely have
access to the quantitative tools that institutional desks use. Reading charts,
tracking RSI/MACD by hand, and deciding when to enter or exit a position is
time-consuming and error-prone, especially for beginners. Most existing
signal tools are either paywalled, opaque ("black box"), or built for markets
outside local investors' reach.

## Solution

TradeSense AI is an end-to-end pipeline that:

1. **Pulls historical price data** for any public ticker (via Yahoo Finance).
2. **Engineers technical-indicator features** — moving averages, RSI, MACD,
   Bollinger Bands, ATR, and momentum features — computed from scratch in
   pandas/numpy (no black-box TA library).
3. **Trains a Random Forest classifier** to predict a forward-looking
   **BUY / HOLD / SELL** signal, validated with walk-forward (time-series)
   cross-validation to avoid look-ahead bias.
4. **Backtests** the resulting strategy against a simple buy-and-hold
   benchmark, reporting return, Sharpe ratio, max drawdown, and trade count.
5. **Serves everything through an interactive Streamlit dashboard** — enter a
   ticker, get a live signal with a confidence score, and see the strategy's
   historical performance.

The model is intentionally interpretable (feature importances are exposed) so
users can see *why* a signal was generated, not just trust a black box.

## Why this approach

- **Random Forest over deep learning**: financial time series are noisy and
  low-signal; a shallow, regularized tree ensemble generalizes better than a
  deep model on a hackathon-scale dataset and trains in seconds, not hours.
- **Hand-rolled indicators**: keeps the dependency footprint small and makes
  every feature auditable — useful for judges reviewing the logic.
- **Walk-forward CV, not random shuffling**: prevents the model from
  "seeing the future," which is the most common mistake in financial ML.
- **Offline-first data loader**: automatically falls back to a realistic
  synthetic price generator when `yfinance`/network access isn't available,
  so the whole pipeline is demoable anywhere, including this repo's own CI
  or a judge's offline machine.

## Project structure

```
TradeSense-AI/
├── app/
│   └── streamlit_app.py     # Interactive dashboard
├── src/
│   ├── data_loader.py       # Yahoo Finance + offline synthetic fallback
│   ├── indicators.py        # SMA/EMA/RSI/MACD/Bollinger/ATR from scratch
│   ├── model.py             # Feature/label building, RF training, inference
│   └── backtest.py          # Long-only backtest vs. buy & hold
├── train.py                 # CLI: run the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## Quickstart

```bash
git clone <this-repo-url>
cd TradeSense-AI
pip install -r requirements.txt

# Train + backtest from the command line
python train.py --symbol AAPL --period 3y

# No internet? Run fully offline on synthetic data
python train.py --symbol AAPL --offline

# Launch the interactive dashboard
streamlit run app/streamlit_app.py
```

## Sample output

```
[3/5] Training RandomForest with walk-forward cross-validation ...
      -> mean CV accuracy: 0.366
[4/5] Backtesting the signal strategy vs. buy & hold ...
      -> strategy return : 94.8%
      -> buy & hold return: -10.09%
      -> max drawdown     : -17.16%
      -> sharpe ratio     : 1.49
[5/5] Generating latest signal + saving model ...
      -> latest signal for AAPL: {'signal': 'HOLD', 'confidence': 0.54, ...}
```

*(Numbers above are from a synthetic offline demo run and will differ with
real market data and different tickers/date ranges.)*

## Roadmap

- Add sentiment features from financial news / social media
- Support multi-asset portfolios and position sizing
- Add LSTM/Transformer model as an alternative predictor for comparison
- Deploy as a hosted web app with saved watchlists per user
- Add risk-adjusted position sizing (Kelly criterion / volatility targeting)

## Disclaimer

TradeSense AI is a hackathon project for educational and research purposes
only. It is **not financial advice**. Past backtest performance does not
guarantee future results.

## Team

Team Lead: _add your name_
Team Members: _add teammates_

## License

MIT

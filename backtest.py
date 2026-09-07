"""
backtest.py
-----------
A minimal long-only backtest: on each BUY signal we go long until the next
SELL signal (or HOLD keeps the current position). This is intentionally
simple and transparent so judges can follow the logic, not a production
execution simulator (no slippage/fees modelling beyond a flat cost).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .model import FEATURE_COLUMNS


def run_backtest(
    model: RandomForestClassifier,
    dataset: pd.DataFrame,
    transaction_cost: float = 0.0005,
) -> dict:
    df = dataset.copy()
    X = df[FEATURE_COLUMNS]
    df["pred"] = model.predict(X)  # 0=SELL, 1=HOLD, 2=BUY

    position = 0  # 0 = flat, 1 = long
    equity = 1.0
    equity_curve = []
    trades = 0

    daily_returns = df["Close"].pct_change().fillna(0)

    for i in range(len(df)):
        signal = df["pred"].iloc[i]

        if signal == 2 and position == 0:  # BUY -> enter
            position = 1
            equity *= 1 - transaction_cost
            trades += 1
        elif signal == 0 and position == 1:  # SELL -> exit
            position = 0
            equity *= 1 - transaction_cost
            trades += 1

        if position == 1:
            equity *= 1 + daily_returns.iloc[i]

        equity_curve.append(equity)

    df["equity_curve"] = equity_curve

    buy_hold_curve = (1 + daily_returns).cumprod()

    total_return = equity_curve[-1] - 1
    buy_hold_return = buy_hold_curve.iloc[-1] - 1

    running_max = pd.Series(equity_curve).cummax()
    drawdown = (pd.Series(equity_curve) - running_max) / running_max
    max_drawdown = drawdown.min()

    daily_strategy_returns = pd.Series(equity_curve).pct_change().fillna(0)
    sharpe = (
        np.sqrt(252) * daily_strategy_returns.mean() / daily_strategy_returns.std()
        if daily_strategy_returns.std() > 0
        else 0.0
    )

    return {
        "total_return_pct": round(total_return * 100, 2),
        "buy_and_hold_return_pct": round(buy_hold_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "num_trades": trades,
        "equity_curve": equity_curve,
        "buy_hold_curve": buy_hold_curve.tolist(),
        "dates": [str(d.date()) for d in df.index],
    }

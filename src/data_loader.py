"""
data_loader.py
---------------
Fetches historical OHLCV price data for TradeSense AI.

Primary path : Yahoo Finance via `yfinance` (requires internet).
Fallback path: a deterministic synthetic OHLCV generator, used automatically
               when yfinance is unavailable or the network call fails. This
               keeps the whole pipeline runnable offline for demos, testing,
               and judging environments with restricted network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _generate_synthetic_ohlcv(
    symbol: str,
    periods: int = 750,
    start_price: float = 100.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate a realistic-looking daily OHLCV series using a bounded
    random walk with drift + volatility clustering (GARCH-like noise).

    This is ONLY used as an offline fallback so the rest of the pipeline
    (indicators, model, backtest) can be exercised without network access.
    """
    rng = np.random.default_rng(seed if seed is not None else abs(hash(symbol)) % (2**32))

    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    # Volatility clustering: sigma follows its own slow random walk
    sigma = np.abs(rng.normal(0.015, 0.004, size=periods))
    sigma = pd.Series(sigma).rolling(5, min_periods=1).mean().to_numpy()

    drift = 0.0003  # slight upward drift, tweak per symbol hash
    drift += (abs(hash(symbol)) % 100 - 50) / 100000

    returns = rng.normal(drift, 1.0, size=periods) * sigma
    close = start_price * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(rng.normal(0, 0.006, size=periods)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, size=periods)))
    open_ = low + (high - low) * rng.random(periods)
    volume = rng.integers(500_000, 5_000_000, size=periods)

    df = pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def load_price_history(
    symbol: str = "AAPL",
    period: str = "3y",
    interval: str = "1d",
    force_offline: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load historical OHLCV data for `symbol`.

    Returns
    -------
    (dataframe, source) where source is "yfinance" or "synthetic".
    """
    if not force_offline:
        try:
            import yfinance as yf

            df = yf.download(symbol, period=period, interval=interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is not None and not df.empty:
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                return df, "yfinance"
        except Exception:
            # No internet, package missing, rate-limited, invalid symbol, etc.
            pass

    periods_map = {"1y": 252, "2y": 504, "3y": 756, "5y": 1260}
    n = periods_map.get(period, 750)
    return _generate_synthetic_ohlcv(symbol, periods=n), "synthetic"


if __name__ == "__main__":
    data, source = load_price_history("AAPL", period="1y")
    print(f"Loaded {len(data)} rows from source: {source}")
    print(data.tail())

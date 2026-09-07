"""
train.py
--------
End-to-end pipeline for TradeSense AI:
    load data -> engineer features -> label -> train -> evaluate -> backtest -> save

Usage:
    python train.py --symbol AAPL --period 3y
    python train.py --symbol AAPL --offline      # force synthetic data, no internet needed
"""

import argparse
import json
import os

from src.backtest import run_backtest
from src.data_loader import load_price_history
from src.model import build_dataset, predict_latest, prepare_features, save_model, train_model


def main():
    parser = argparse.ArgumentParser(description="Train the TradeSense AI signal model")
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol, e.g. AAPL, MSFT, TSLA")
    parser.add_argument("--period", default="3y", help="History window: 1y, 2y, 3y, 5y")
    parser.add_argument("--horizon", type=int, default=5, help="Forward-looking days for labeling")
    parser.add_argument("--threshold", type=float, default=0.015, help="Return threshold for BUY/SELL")
    parser.add_argument("--offline", action="store_true", help="Force synthetic data (skip yfinance)")
    args = parser.parse_args()

    print(f"[1/5] Loading price history for {args.symbol} ({args.period}) ...")
    raw_df, source = load_price_history(args.symbol, period=args.period, force_offline=args.offline)
    print(f"      -> {len(raw_df)} rows loaded (source: {source})")

    print("[2/5] Engineering technical-indicator features + labels ...")
    dataset = build_dataset(raw_df, horizon=args.horizon, threshold=args.threshold)
    print(f"      -> {len(dataset)} labeled rows ready for training")
    print(f"      -> label distribution: {dataset['label'].value_counts().to_dict()}")

    print("[3/5] Training RandomForest with walk-forward cross-validation ...")
    model, metrics = train_model(dataset)
    print(f"      -> mean CV accuracy: {metrics['cv_mean_accuracy']:.3f}")
    print("      -> top features:", list(metrics["feature_importances"].items())[:5])

    print("[4/5] Backtesting the signal strategy vs. buy & hold ...")
    bt = run_backtest(model, dataset)
    print(f"      -> strategy return : {bt['total_return_pct']}%")
    print(f"      -> buy & hold return: {bt['buy_and_hold_return_pct']}%")
    print(f"      -> max drawdown     : {bt['max_drawdown_pct']}%")
    print(f"      -> sharpe ratio     : {bt['sharpe_ratio']}")
    print(f"      -> number of trades : {bt['num_trades']}")

    print("[5/5] Generating latest signal + saving model ...")
    live_features = prepare_features(raw_df)
    latest_signal = predict_latest(model, live_features)
    print(f"      -> latest signal for {args.symbol}: {latest_signal}")

    os.makedirs("models", exist_ok=True)
    save_model(model, f"models/tradesense_{args.symbol.lower()}.joblib")

    summary = {
        "symbol": args.symbol,
        "data_source": source,
        "rows_used": len(dataset),
        "metrics": {
            "cv_mean_accuracy": metrics["cv_mean_accuracy"],
            "cv_fold_accuracies": metrics["cv_fold_accuracies"],
            "top_features": list(metrics["feature_importances"].items())[:5],
        },
        "backtest": {k: v for k, v in bt.items() if k not in ("equity_curve", "buy_hold_curve", "dates")},
        "latest_signal": latest_signal,
    }
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{args.symbol.lower()}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved model -> models/tradesense_{args.symbol.lower()}.joblib")
    print(f"Saved summary -> outputs/{args.symbol.lower()}_summary.json")


if __name__ == "__main__":
    main()

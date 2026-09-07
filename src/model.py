"""
model.py
--------
Builds the supervised learning problem for TradeSense AI:

  Label:  looking `horizon` days ahead, was the forward return
          > +threshold  -> BUY
          < -threshold  -> SELL
          otherwise      -> HOLD

  Model:  RandomForestClassifier over the engineered technical-indicator
          feature set from indicators.py.

Chosen for interpretability (feature_importances_), robustness to noisy
financial features, and fast train/predict without GPU requirements —
appropriate for a hackathon-scale demo.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

from .indicators import add_all_indicators

FEATURE_COLUMNS = [
    "sma_10",
    "sma_50",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_pct",
    "atr_14",
    "pct_change_1d",
    "pct_change_5d",
    "volume_change",
]

LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


def prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicator features and drop only the warm-up rows that lack
    them (used for both training and live inference on the newest row)."""
    df = add_all_indicators(raw_df)
    return df.dropna(subset=FEATURE_COLUMNS)


def build_dataset(
    raw_df: pd.DataFrame, horizon: int = 5, threshold: float = 0.015
) -> pd.DataFrame:
    """Turn raw OHLCV data into a feature + label dataframe for TRAINING.

    Note: the most recent `horizon` rows have no known forward return yet
    and are dropped here. For live inference on the latest bar, use
    `prepare_features()` directly instead.
    """
    df = add_all_indicators(raw_df)
    forward_return = df["Close"].shift(-horizon) / df["Close"] - 1

    conditions = [forward_return > threshold, forward_return < -threshold]
    choices = [2, 0]  # BUY, SELL
    df["label"] = np.select(conditions, choices, default=1)  # HOLD

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    return df


def train_model(
    dataset: pd.DataFrame, n_splits: int = 5
) -> tuple[RandomForestClassifier, dict]:
    """Train a RandomForestClassifier with walk-forward (time series) CV."""
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_reports = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], preds)
        fold_reports.append(acc)

    # Final model trained on the full dataset for deployment/inference
    final_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
    )
    final_model.fit(X, y)

    final_preds = final_model.predict(X)
    report = classification_report(
        y, final_preds, target_names=["SELL", "HOLD", "BUY"], output_dict=True
    )

    metrics = {
        "cv_fold_accuracies": fold_reports,
        "cv_mean_accuracy": float(np.mean(fold_reports)),
        "train_classification_report": report,
        "feature_importances": dict(
            sorted(
                zip(FEATURE_COLUMNS, final_model.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }
    return final_model, metrics


def predict_latest(model: RandomForestClassifier, dataset: pd.DataFrame) -> dict:
    """Predict a signal for the most recent row in `dataset`."""
    latest = dataset[FEATURE_COLUMNS].iloc[[-1]]
    pred = int(model.predict(latest)[0])
    proba = model.predict_proba(latest)[0]
    return {
        "signal": LABEL_MAP[pred],
        "confidence": float(np.max(proba)),
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(proba)},
        "as_of": str(dataset.index[-1].date()),
    }


def save_model(model: RandomForestClassifier, path: str = "models/tradesense_rf.joblib"):
    joblib.dump(model, path)


def load_model(path: str = "models/tradesense_rf.joblib") -> RandomForestClassifier:
    return joblib.load(path)

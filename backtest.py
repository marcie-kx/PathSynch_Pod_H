#!/usr/bin/env python3
from __future__ import annotations

"""Run recursive rolling backtests for a selected target.

Key behavior:
- Trains on expanding history.
- Predicts each fold horizon recursively (multi-step), not one-shot X_test scoring.
- Stores fold metrics and absolute errors for uncertainty calibration.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from utils import TARGET_CONFIGS, aggregate_ad_metrics, build_xgb_model, compute_recency_weights, ensure_dir, inverse_transform, raw_data_path, recursive_forecast, root_path


def sliding_backtest(
    df: pd.DataFrame,
    config,
    ad_metrics: dict[str, pd.Series],
    fold_size: int = 30,
    n_folds: int = 4,
    recency_half_life_days: float | None = None,
) -> list[dict]:
    """Evaluate a model with expanding-window folds and recursive rollout.

    Important detail:
    - Features are generated during rollout from raw-scale history.
    - Model predictions remain in transformed target space and are inverse-transformed
      only for metric computation and error storage.
    """
    results = []
    rows = len(df)
    for fold in range(n_folds):
        train_end = rows - (n_folds - fold) * fold_size
        test_end = train_end + fold_size
        if test_end > rows:
            break

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        X_train = train_df.drop(columns=["target"])
        y_train = train_df["target"].values
        sample_weight = compute_recency_weights(train_df.index, recency_half_life_days)

        model = build_xgb_model(config)
        model.fit(X_train, y_train, sample_weight=sample_weight)

        # Recursive rollout uses raw-scale history for lag/rolling consistency.
        history_raw = pd.Series(
            inverse_transform(train_df["target"].values, config.transform),
            index=train_df.index,
        ).astype(float)
        pred = recursive_forecast(model, history_raw, test_df.index, config, ad_metrics)

        y_test = test_df["target"].values

        test_pred = inverse_transform(pred, config.transform)
        test_true = inverse_transform(y_test, config.transform)
        mape = float(mean_absolute_percentage_error(test_true, test_pred) * 100)
        within_10 = float(np.mean(np.abs((test_true - test_pred) / np.maximum(np.abs(test_true), 1e-6)) * 100 <= 10) * 100)

        results.append(
            {
                "fold": fold + 1,
                "train_start": str(train_df.index.min()),
                "train_end": str(train_df.index.max()),
                "test_start": str(test_df.index.min()),
                "test_end": str(test_df.index.max()),
                "mape": mape,
                "within_10": within_10,
                "n_test": len(test_df),
                "abs_errors": np.abs(test_true - test_pred).tolist(),
            }
        )
    return results


def run_backtest(target: str, recency_half_life_days: float | None = None) -> None:
    """Entry point for one target backtest and metric persistence."""
    config = TARGET_CONFIGS[target]
    raw = pd.read_csv(raw_data_path())
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")
    raw = raw.dropna(subset=["Transaction_Date"])
    ad_metrics = aggregate_ad_metrics(raw, "Transaction_Date")

    feature_path = root_path() / "projects" / "synthetic_ecommerce_data" / "data" / "processed" / f"{target.lower()}_features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.sort_index()

    if len(df) < 120:
        raise ValueError("Not enough rows for backtest. At least 120 are recommended.")

    backtest_results = sliding_backtest(
        df,
        config,
        ad_metrics,
        recency_half_life_days=recency_half_life_days,
    )
    summary = {
        "target": target,
        "folds": len(backtest_results),
        "avg_mape": float(np.mean([r["mape"] for r in backtest_results])),
        "avg_within_10": float(np.mean([r["within_10"] for r in backtest_results])),
        "recency_half_life_days": recency_half_life_days,
        # Used later by forecast_30d.py to build 85% prediction intervals.
        "abs_error_quantile_925": float(np.quantile([err for fold in backtest_results for err in fold["abs_errors"]], 0.925)),
    }

    out_dir = root_path() / "projects" / "synthetic_ecommerce_data" / "outputs" / "backtests" / target.lower()
    ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps({"summary": summary, "folds": backtest_results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Backtest complete for {target}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling CV backtest for a target.")
    parser.add_argument("--target", type=str, default="Revenue")
    parser.add_argument("--recency-half-life", type=float, default=None)
    args = parser.parse_args()
    run_backtest(args.target, recency_half_life_days=args.recency_half_life)

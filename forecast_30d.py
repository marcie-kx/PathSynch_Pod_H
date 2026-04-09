#!/usr/bin/env python3
from __future__ import annotations

"""Train final model and generate a 30-day recursive forecast.

Highlights:
- Uses the same recursive mechanics as backtest for consistency.
- Builds 85% intervals from backtest out-of-fold error quantiles when available.
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from utils import TARGET_CONFIGS, aggregate_daily, aggregate_ad_metrics, build_xgb_model, compute_recency_weights, ensure_dir, inverse_transform, raw_data_path, recursive_forecast, root_path


def run_forecast(target: str, recency_half_life_days: float | None = None) -> None:
    """Train on full history and forecast next 30 daily points."""
    config = TARGET_CONFIGS[target]
    raw = pd.read_csv(raw_data_path())
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")
    raw = raw.dropna(subset=["Transaction_Date"])

    daily_target = aggregate_daily(raw, "Transaction_Date", target)
    ad_metrics = aggregate_ad_metrics(raw, "Transaction_Date")

    feature_path = root_path() / "projects" / "synthetic_ecommerce_data" / "data" / "processed" / f"{target.lower()}_features.parquet"
    df = pd.read_parquet(feature_path)
    df = df.sort_index()

    X_train = df.drop(columns=["target"])
    y_train = df["target"].values
    sample_weight = compute_recency_weights(df.index, recency_half_life_days)

    model = build_xgb_model(config)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    last_date = daily_target.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")

    # Keep autoregressive feature generation on raw scale during rollout.
    history_raw = pd.Series(
        inverse_transform(df["target"].values, config.transform),
        index=df.index,
    ).astype(float)
    preds_transformed = recursive_forecast(model, history_raw, future_dates, config, ad_metrics)
    preds = inverse_transform(preds_transformed, config.transform)

    metrics_path = root_path() / "projects" / "synthetic_ecommerce_data" / "outputs" / "backtests" / target.lower() / "metrics.json"
    quantile_85 = None
    if metrics_path.exists():
        # Preferred interval source: out-of-fold absolute errors from backtest.
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        quantile_85 = float(metrics.get("summary", {}).get("abs_error_quantile_925", 0.0))

    if quantile_85 is None or quantile_85 <= 0:
        # Fallback when backtest metrics are unavailable.
        train_preds_transformed = model.predict(X_train)
        train_preds = inverse_transform(train_preds_transformed, config.transform)
        y_train_original = inverse_transform(y_train, config.transform)
        residuals = np.abs(y_train_original - train_preds)
        quantile_85 = float(np.quantile(residuals, 0.925))

    lower_bounds = np.maximum(preds - quantile_85, 0)
    upper_bounds = preds + quantile_85

    out_dir = root_path() / "projects" / "synthetic_ecommerce_data" / "outputs" / "forecasts" / target.lower()
    ensure_dir(out_dir)
    forecast_df = pd.DataFrame({
        "date": future_dates,
        "prediction": preds,
        "lower_85": lower_bounds,
        "upper_85": upper_bounds
    })
    forecast_df.to_csv(out_dir / "forecast_30d.csv", index=False)
    print(f"Saved 30-day forecast with 85% CI for {target} to {out_dir / 'forecast_30d.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train final XGBoost model and forecast next 30 days.")
    parser.add_argument("--target", type=str, default="Revenue")
    parser.add_argument("--recency-half-life", type=float, default=None)
    args = parser.parse_args()
    run_forecast(args.target, recency_half_life_days=args.recency_half_life)

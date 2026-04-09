#!/usr/bin/env python3
from __future__ import annotations

"""Build daily supervised feature tables for each target.

Why this file exists:
- Converts raw transactional rows into daily modeling tables.
- Produces a consistent feature schema used by backtest and forecast scripts.

What it writes:
- projects/synthetic_ecommerce_data/data/processed/revenue_features.parquet
- projects/synthetic_ecommerce_data/data/processed/clicks_features.parquet
- projects/synthetic_ecommerce_data/data/processed/conversion_rate_features.parquet
"""

from pathlib import Path
import pandas as pd

from utils import TARGET_CONFIGS, aggregate_daily, aggregate_ad_metrics, build_feature_matrix, ensure_dir, raw_data_path, root_path


def main() -> None:
    """End-to-end feature generation for all configured targets.

    Steps:
    1) Load and clean raw data.
    2) Aggregate exogenous ad metrics once at daily level.
    3) Build target-specific supervised feature matrices.
    4) Save one parquet file per target.
    """
    raw_path = raw_data_path()
    raw = pd.read_csv(raw_path)
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")
    raw = raw.dropna(subset=["Transaction_Date"])

    output_dir = root_path() / "projects" / "synthetic_ecommerce_data" / "data" / "processed"
    ensure_dir(output_dir)

    ad_metrics = aggregate_ad_metrics(raw, "Transaction_Date")

    for target_name, config in TARGET_CONFIGS.items():
        # Build each target independently so model settings and transforms can differ.
        daily_target = aggregate_daily(raw, "Transaction_Date", target_name)
        feature_dates = daily_target.index[1:]
        features = build_feature_matrix(feature_dates, daily_target, config, ad_metrics)
        features = features.dropna(axis=0, how="any")
        features.to_parquet(output_dir / f"{target_name.lower()}_features.parquet", index=True)
        print(f"Saved features for {target_name}: {len(features)} rows")

    print("Feature building complete")


if __name__ == "__main__":
    main()

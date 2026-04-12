#!/usr/bin/env python3
from __future__ import annotations

"""Deal Recommendations: single production recommender for Revenue.

End-to-end approach
1) Train Revenue XGBoost on full history using the existing feature table.
2) Build a baseline 30-day recursive forecast.
3) Simulate multiple Ad_Spend scenarios (light/standard/aggressive/heavy).
4) Estimate candidate quality using three scoring modes:
    - growth_score: prioritize uplift
    - efficiency_score: prioritize profit/ROI proxy
    - balanced_score: compromise between growth and efficiency
5) Export top recommendations and decomposition summaries.

Core proxy calculations
- uplift = scenario_revenue - baseline_revenue
- incremental_profit_proxy = uplift - ad_cost_weight * extra_ad_cost
- roi_proxy = incremental_profit_proxy / extra_ad_cost

Important constraints
- Scenario based: results depend on assumed Ad_Spend multipliers.
- Day-level only: no promo-type or intraday optimization.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    TARGET_CONFIGS,
    aggregate_ad_metrics,
    aggregate_daily,
    build_xgb_model,
    compute_recency_weights,
    ensure_dir,
    inverse_transform,
    raw_data_path,
    recursive_forecast,
    root_path,
)


SCENARIO_LIBRARY: dict[str, float] = {
    "light_push": 1.25,
    "standard_push": 1.50,
    "aggressive_push": 2.00,
    "heavy_push": 2.50,
}


def minmax_norm(values: pd.Series) -> pd.Series:
    """Normalize a numeric series to [0, 1]; return 0.5 when all values are identical."""
    if values.empty:
        return pd.Series(dtype=float)

    min_value = float(values.min())
    max_value = float(values.max())
    if np.isclose(min_value, max_value):
        return pd.Series(0.5, index=values.index, dtype=float)

    return ((values - min_value) / (max_value - min_value)).astype(float)


def effective_max_interval_width_pct(
    dates: pd.Series,
    max_interval_width_pct: float,
    year_end_relaxation: float,
) -> pd.Series:
    """Return date-specific uncertainty thresholds with Nov/Dec relaxation."""
    dt = pd.to_datetime(dates)
    is_year_end = dt.dt.month.isin([11, 12])
    return pd.Series(
        np.where(
            is_year_end,
            max_interval_width_pct + year_end_relaxation,
            max_interval_width_pct,
        ),
        index=dates.index,
        dtype=float,
    )


def extend_ad_metrics_for_future(
    ad_metrics: dict[str, pd.Series],
    future_dates: pd.DatetimeIndex,
    ad_spend_multiplier: float,
) -> dict[str, pd.Series]:
    """Extend daily ad metrics into the forecast horizon.

    Future Ad_Spend is multiplied by the provided scenario multiplier.
    Ad_CTR and Ad_CPC remain unchanged unless future logic changes later.
    """
    extended: dict[str, pd.Series] = {}
    for metric_name, series in ad_metrics.items():
        base_series = series.astype(float).sort_index()
        if base_series.empty:
            raise ValueError(f"Ad metric series is empty: {metric_name}")

        last_value = float(base_series.iloc[-1])
        future_values = pd.Series(last_value, index=future_dates, dtype=float)
        if metric_name == "Ad_Spend":
            future_values = future_values * float(ad_spend_multiplier)

        extended_series = pd.concat([base_series, future_values]).sort_index()
        extended[metric_name] = extended_series.astype(float)

    return extended


def load_interval_half_width(
    metrics_path: Path,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config,
) -> float:
    """Load the 85% interval half-width from backtest output, with a training fallback."""
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        half_width = float(metrics.get("summary", {}).get("abs_error_quantile_925", 0.0))
        if half_width > 0:
            return half_width

    # Fallback mirrors the logic used in forecast_30d.py.
    train_preds_transformed = model.predict(X_train)
    train_preds = inverse_transform(train_preds_transformed, config.transform)
    y_train_original = inverse_transform(y_train, config.transform)
    residuals = np.abs(y_train_original - train_preds)
    return float(np.quantile(residuals, 0.925))


def build_baseline_and_scenario_forecasts(
    model,
    history_raw: pd.Series,
    future_dates: pd.DatetimeIndex,
    config,
    baseline_ad_metrics: dict[str, pd.Series],
    scenario_ad_metrics: dict[str, pd.Series],
    interval_half_width: float,
) -> pd.DataFrame:
    """Create baseline/scenario forecasts and baseline confidence intervals."""
    baseline_pred_t = recursive_forecast(model, history_raw, future_dates, config, baseline_ad_metrics)
    scenario_pred_t = recursive_forecast(model, history_raw, future_dates, config, scenario_ad_metrics)

    baseline_revenue = inverse_transform(baseline_pred_t, config.transform)
    scenario_revenue = inverse_transform(scenario_pred_t, config.transform)

    lower_85 = np.maximum(baseline_revenue - interval_half_width, 0)
    upper_85 = baseline_revenue + interval_half_width
    interval_width = upper_85 - lower_85
    interval_width_pct = interval_width / np.maximum(np.abs(baseline_revenue), 1e-6)

    return pd.DataFrame(
        {
            "date": future_dates.date.astype(str),
            "weekday": future_dates.day_name(),
            "baseline_revenue": baseline_revenue.astype(float),
            "scenario_revenue": scenario_revenue.astype(float),
            "uplift": (scenario_revenue - baseline_revenue).astype(float),
            "uplift_pct": ((scenario_revenue - baseline_revenue) / np.maximum(np.abs(baseline_revenue), 1e-6) * 100.0).astype(float),
            "lower_85": lower_85.astype(float),
            "upper_85": upper_85.astype(float),
            "interval_width": interval_width.astype(float),
            "interval_width_pct": interval_width_pct.astype(float),
        }
    )


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add growth, efficiency, and balanced scores to a filtered candidate frame."""
    if df.empty:
        out = df.copy()
        out["growth_score"] = pd.Series(dtype=float)
        out["efficiency_score"] = pd.Series(dtype=float)
        out["balanced_score"] = pd.Series(dtype=float)
        return out

    out = df.copy()
    uplift_norm = minmax_norm(out["uplift"])
    uplift_pct_norm = minmax_norm(out["uplift_pct"])
    width_norm = minmax_norm(out["interval_width_pct"])
    profit_norm = minmax_norm(out["incremental_profit_proxy"])
    roi_norm = minmax_norm(out["roi_proxy"])

    out["growth_score"] = (0.60 * uplift_norm + 0.20 * uplift_pct_norm - 0.20 * width_norm).astype(float)
    out["efficiency_score"] = (0.45 * profit_norm + 0.35 * roi_norm - 0.20 * width_norm).astype(float)
    out["balanced_score"] = (
        0.35 * uplift_norm
        + 0.25 * profit_norm
        + 0.20 * roi_norm
        + 0.10 * uplift_pct_norm
        - 0.10 * width_norm
    ).astype(float)
    return out


def add_reason_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach transparent primary/secondary reason tags for recommendation rows."""
    out = df.copy()
    if out.empty:
        out["primary_reason"] = pd.Series(dtype=str)
        out["secondary_reason"] = pd.Series(dtype=str)
        return out

    low_uncertainty_cut = float(out["interval_width_pct"].quantile(0.25))
    weekday_profit = out.groupby("weekday", as_index=False)["incremental_profit_proxy"].mean()
    best_weekday = str(weekday_profit.sort_values("incremental_profit_proxy", ascending=False).iloc[0]["weekday"])
    eff_strong_cut = float(out["efficiency_score"].quantile(0.70))

    primary: list[str] = []
    secondary: list[str] = []

    for _, row in out.iterrows():
        growth = float(row["growth_score"])
        eff = float(row["efficiency_score"])
        bal = float(row["balanced_score"])
        month = int(pd.to_datetime(row["date"]).month)
        weekday = str(row["weekday"])
        scenario = str(row["scenario_name"])
        interval_pct = float(row["interval_width_pct"])

        if growth >= eff and growth >= bal:
            pri = "High growth candidate"
        elif eff >= growth and eff >= bal:
            pri = "High efficiency candidate"
        else:
            pri = "Balanced candidate"

        sec = ""
        if interval_pct <= low_uncertainty_cut:
            sec = "Stable low-uncertainty opportunity"
        if month in (11, 12):
            sec = "Year-end seasonal opportunity"
        if weekday == "Saturday" and best_weekday == "Saturday":
            sec = "Strong Saturday pattern"
        if scenario == "light_push" and eff >= eff_strong_cut:
            sec = "Moderate spend efficiency"

        primary.append(pri)
        secondary.append(sec)

    out["primary_reason"] = primary
    out["secondary_reason"] = secondary
    return out


def build_weekday_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekday decomposition metrics for retained candidates."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "weekday",
                "avg_uplift",
                "avg_incremental_profit_proxy",
                "avg_roi_proxy",
                "recommended_days_count",
            ]
        )
    return (
        df.groupby("weekday", as_index=False)
        .agg(
            avg_uplift=("uplift", "mean"),
            avg_incremental_profit_proxy=("incremental_profit_proxy", "mean"),
            avg_roi_proxy=("roi_proxy", "mean"),
            recommended_days_count=("date", "count"),
        )
        .sort_values("avg_incremental_profit_proxy", ascending=False)
        .reset_index(drop=True)
    )


def build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scenario decomposition metrics for retained candidates."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "scenario_name",
                "avg_uplift",
                "avg_incremental_profit_proxy",
                "avg_roi_proxy",
                "recommended_days_count",
            ]
        )
    return (
        df.groupby("scenario_name", as_index=False)
        .agg(
            avg_uplift=("uplift", "mean"),
            avg_incremental_profit_proxy=("incremental_profit_proxy", "mean"),
            avg_roi_proxy=("roi_proxy", "mean"),
            recommended_days_count=("date", "count"),
        )
        .sort_values("avg_incremental_profit_proxy", ascending=False)
        .reset_index(drop=True)
    )


def filter_candidates(
    df: pd.DataFrame,
    min_uplift_pct: float,
    max_interval_width_pct: float,
    year_end_relaxation: float,
) -> pd.DataFrame:
    """Apply candidate filters before scoring."""
    out = df.copy()
    out["effective_max_interval_width_pct"] = effective_max_interval_width_pct(
        out["date"],
        max_interval_width_pct=max_interval_width_pct,
        year_end_relaxation=year_end_relaxation,
    )
    return out.loc[
        (out["uplift"] > 0)
        & (out["uplift_pct"] >= min_uplift_pct * 100.0)
        & (out["interval_width_pct"] <= out["effective_max_interval_width_pct"])
    ].copy()


def run_recommendations(
    target: str,
    horizon: int,
    top_n: int,
    min_uplift_pct: float,
    max_interval_width_pct: float,
    year_end_relaxation: float,
    ad_cost_weight: float,
    recency_half_life_days: float | None = None,
) -> None:
    """Run the full recommendation pipeline and save outputs."""
    if target != "Revenue":
        raise ValueError("This recommender is designed for Revenue only.")

    config = TARGET_CONFIGS[target]

    raw = pd.read_csv(raw_data_path())
    raw["Transaction_Date"] = pd.to_datetime(raw["Transaction_Date"], errors="coerce")
    raw = raw.dropna(subset=["Transaction_Date"])

    daily_target = aggregate_daily(raw, "Transaction_Date", target)
    ad_metrics = aggregate_ad_metrics(raw, "Transaction_Date")

    feature_path = root_path() / "data" / "processed" / f"{target.lower()}_features.parquet"
    df = pd.read_parquet(feature_path).sort_index()

    X_train = df.drop(columns=["target"])
    y_train = df["target"].values
    sample_weight = compute_recency_weights(df.index, recency_half_life_days)

    model = build_xgb_model(config)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    last_date = daily_target.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    history_raw = pd.Series(
        inverse_transform(df["target"].values, config.transform),
        index=df.index,
    ).astype(float)

    baseline_ad_metrics = extend_ad_metrics_for_future(ad_metrics, future_dates, ad_spend_multiplier=1.0)
    baseline_spend_future = baseline_ad_metrics["Ad_Spend"].reindex(future_dates).astype(float)

    backtest_metrics_path = root_path() / "outputs" / "backtests" / target.lower() / "metrics.json"
    interval_half_width = load_interval_half_width(backtest_metrics_path, model, X_train, y_train, config)

    all_scenarios: list[pd.DataFrame] = []
    for scenario_name, mult in SCENARIO_LIBRARY.items():
        scenario_ad_metrics = extend_ad_metrics_for_future(ad_metrics, future_dates, ad_spend_multiplier=mult)
        scenario_spend_future = scenario_ad_metrics["Ad_Spend"].reindex(future_dates).astype(float)

        scenario_df = build_baseline_and_scenario_forecasts(
            model=model,
            history_raw=history_raw,
            future_dates=future_dates,
            config=config,
            baseline_ad_metrics=baseline_ad_metrics,
            scenario_ad_metrics=scenario_ad_metrics,
            interval_half_width=interval_half_width,
        )

        scenario_df["scenario_name"] = scenario_name
        scenario_df["ad_spend_multiplier"] = float(mult)
        scenario_df["future_baseline_ad_spend"] = baseline_spend_future.values
        scenario_df["future_scenario_ad_spend"] = scenario_spend_future.values
        scenario_df["extra_ad_cost"] = (scenario_spend_future - baseline_spend_future).values
        scenario_df["incremental_profit_proxy"] = scenario_df["uplift"] - ad_cost_weight * scenario_df["extra_ad_cost"]
        scenario_df["roi_proxy"] = scenario_df["incremental_profit_proxy"] / np.maximum(scenario_df["extra_ad_cost"], 1e-6)

        all_scenarios.append(scenario_df)

    all_candidates = pd.concat(all_scenarios, ignore_index=True)
    filtered = filter_candidates(
        all_candidates,
        min_uplift_pct=min_uplift_pct,
        max_interval_width_pct=max_interval_width_pct,
        year_end_relaxation=year_end_relaxation,
    )
    scored = add_scores(filtered)
    recommend_df = add_reason_tags(scored)
    recommend_df["recommend_flag"] = True

    top_growth = recommend_df.sort_values(["growth_score", "uplift"], ascending=[False, False]).head(top_n).reset_index(drop=True)
    top_efficiency = recommend_df.sort_values(
        ["efficiency_score", "incremental_profit_proxy"], ascending=[False, False]
    ).head(top_n).reset_index(drop=True)
    top_balanced = recommend_df.sort_values(["balanced_score", "uplift"], ascending=[False, False]).head(top_n).reset_index(drop=True)

    weekday_summary = build_weekday_summary(recommend_df)
    scenario_summary = build_scenario_summary(recommend_df)

    out_dir = root_path() / "outputs" / "recommendations" / target.lower()
    ensure_dir(out_dir)

    full_columns = [
        "date",
        "weekday",
        "scenario_name",
        "ad_spend_multiplier",
        "baseline_revenue",
        "scenario_revenue",
        "uplift",
        "uplift_pct",
        "future_baseline_ad_spend",
        "future_scenario_ad_spend",
        "extra_ad_cost",
        "incremental_profit_proxy",
        "roi_proxy",
        "lower_85",
        "upper_85",
        "interval_width",
        "interval_width_pct",
        "effective_max_interval_width_pct",
        "growth_score",
        "efficiency_score",
        "balanced_score",
        "primary_reason",
        "secondary_reason",
        "recommend_flag",
    ]

    recommend_df = recommend_df.reindex(columns=full_columns)
    top_growth = top_growth.reindex(columns=full_columns)
    top_efficiency = top_efficiency.reindex(columns=full_columns)
    top_balanced = top_balanced.reindex(columns=full_columns)

    recommend_df.to_csv(out_dir / "deal_recommendations_full.csv", index=False)
    top_growth.to_csv(out_dir / "deal_recommendations_top_growth.csv", index=False)
    top_efficiency.to_csv(out_dir / "deal_recommendations_top_efficiency.csv", index=False)
    top_balanced.to_csv(out_dir / "deal_recommendations_top_balanced.csv", index=False)
    weekday_summary.to_csv(out_dir / "deal_weekday_decomposition.csv", index=False)
    scenario_summary.to_csv(out_dir / "deal_scenario_decomposition.csv", index=False)

    metadata = {
        "target": target,
        "horizon_days": horizon,
        "scenario_names": list(SCENARIO_LIBRARY.keys()),
        "min_uplift_pct": min_uplift_pct,
        "max_interval_width_pct": max_interval_width_pct,
        "year_end_relaxation": year_end_relaxation,
        "ad_cost_weight": ad_cost_weight,
        "top_n": top_n,
        "recency_half_life_days": recency_half_life_days,
        "rows_considered": int(len(all_candidates)),
        "rows_recommended": int(len(recommend_df)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Deal recommendations run complete")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved: {out_dir / 'deal_recommendations_full.csv'}")
    print(f"Saved: {out_dir / 'deal_recommendations_top_growth.csv'}")
    print(f"Saved: {out_dir / 'deal_recommendations_top_efficiency.csv'}")
    print(f"Saved: {out_dir / 'deal_recommendations_top_balanced.csv'}")
    print(f"Saved: {out_dir / 'deal_weekday_decomposition.csv'}")
    print(f"Saved: {out_dir / 'deal_scenario_decomposition.csv'}")
    print(f"Saved: {out_dir / 'metadata.json'}")


def main() -> None:
    """Parse CLI arguments and execute the recommender."""
    parser = argparse.ArgumentParser(description="Deal Recommendations for Revenue (day-level multi-scenario).")
    parser.add_argument("--target", type=str, default="Revenue")
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--min-uplift-pct", type=float, default=0.003)
    parser.add_argument("--max-interval-width-pct", type=float, default=0.27)
    parser.add_argument("--year-end-relaxation", type=float, default=0.15)
    parser.add_argument("--ad-cost-weight", type=float, default=0.1)
    parser.add_argument("--recency-half-life", type=float, default=None)
    args = parser.parse_args()

    run_recommendations(
        target=args.target,
        horizon=args.horizon,
        top_n=args.top_n,
        min_uplift_pct=args.min_uplift_pct,
        max_interval_width_pct=args.max_interval_width_pct,
        year_end_relaxation=args.year_end_relaxation,
        ad_cost_weight=args.ad_cost_weight,
        recency_half_life_days=args.recency_half_life,
    )


if __name__ == "__main__":
    main()

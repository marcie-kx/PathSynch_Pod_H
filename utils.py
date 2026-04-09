#!/usr/bin/env python3
from __future__ import annotations

"""Shared utilities for the team delivery XGBoost pipeline.

Design intent:
- Keep all target configs in one place.
- Centralize feature engineering so train/backtest/forecast stay consistent.
- Provide recursive rollout that preserves raw-scale autoregressive history.

Important assumption:
- Ad metrics are treated as known exogenous/scenario inputs.
"""

from dataclasses import dataclass
from pathlib import Path
import json

import holidays
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class TargetConfig:
    """Container for per-target modeling rules and hyperparameters."""
    name: str
    aggregation: str
    transform: str
    clip_min: float | None
    clip_max: float | None
    lags: tuple[int, ...]
    rolling_windows: tuple[int, ...]
    model_params: dict


TARGET_CONFIGS: dict[str, TargetConfig] = {
    # Revenue is the production target and uses the latest tuned params.
    "Revenue": TargetConfig(
        name="Revenue",
        aggregation="sum",
        transform="log1p",
        clip_min=0.0,
        clip_max=None,
        lags=(1, 2, 3, 7, 14, 21, 28, 35, 56),
        rolling_windows=(3, 7, 14, 28),
        model_params={
            "n_estimators": 300,
            "learning_rate": 0.015,
            "max_depth": 2,
            "min_child_weight": 6,
            "subsample": 1.0,
            "colsample_bytree": 0.6,
            "reg_alpha": 0.0,
            "reg_lambda": 1.5,
            "random_state": 42,
        },
    ),
    "Clicks": TargetConfig(
        name="Clicks",
        aggregation="sum",
        transform="log1p",
        clip_min=0.0,
        clip_max=None,
        lags=(1, 2, 3, 7, 14, 21, 28, 35, 56),
        rolling_windows=(3, 7, 14, 28),
        model_params={
            "n_estimators": 700,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "random_state": 42,
        },
    ),
    "Conversion_Rate": TargetConfig(
        name="Conversion_Rate",
        aggregation="mean",
        transform="log1p",
        clip_min=0.0,
        clip_max=None,
        lags=(1, 2, 3, 7, 14, 21, 28),
        rolling_windows=(3, 7, 14, 28),
        model_params={
            "n_estimators": 500,
            "learning_rate": 0.04,
            "max_depth": 3,
            "min_child_weight": 5,
            "subsample": 0.95,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "random_state": 42,
        },
    ),
}


def root_path() -> Path:
    """Return repository root from the team delivery script location."""
    return Path(__file__).resolve().parents[2]


def raw_data_path() -> Path:
    """Return canonical raw CSV path used by all scripts."""
    return root_path() / "projects" / "synthetic_ecommerce_data" / "data" / "raw" / "synthetic_ecommerce_data.csv"


def ensure_dir(path: Path) -> None:
    """Create directory recursively if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def transform_target(values: np.ndarray | pd.Series, transform: str) -> np.ndarray:
    """Apply configured target transform before model training."""
    if transform == "identity":
        return np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.log1p(np.asarray(values, dtype=float))
    if transform == "logit":
        clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(clipped / (1 - clipped))
    raise ValueError(f"Unsupported transform: {transform}")


def inverse_transform(values: np.ndarray | pd.Series, transform: str) -> np.ndarray:
    """Map model outputs back to the original target scale."""
    if transform == "identity":
        return np.asarray(values, dtype=float)
    if transform == "log1p":
        arr = np.asarray(values, dtype=float)
        arr = np.clip(arr, -700, 700)  # Prevent overflow in expm1
        return np.expm1(arr)
    if transform == "logit":
        arr = np.asarray(values, dtype=float)
        return 1.0 / (1.0 + np.exp(-arr))
    raise ValueError(f"Unsupported transform: {transform}")


def aggregate_daily(df: pd.DataFrame, date_col: str, target_col: str) -> pd.Series:
    """Aggregate a target into daily series according to target config."""
    config = TARGET_CONFIGS[target_col]
    temp = df[[date_col, target_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, target_col])
    # The aggregation method is target-dependent (sum for Revenue/Clicks, mean for Conversion_Rate).
    grouped = getattr(temp.groupby(date_col, as_index=True)[target_col], config.aggregation)()
    grouped.index = pd.to_datetime(grouped.index)
    return grouped.sort_index().astype(float)


def aggregate_ad_metrics(df: pd.DataFrame, date_col: str) -> dict[str, pd.Series]:
    """Build daily ad metric series in modeling-friendly form.

    Conventions:
    - Ad_Spend: daily sum
    - Ad_CTR: daily clicks / daily impressions
    - Ad_CPC: daily spend / daily clicks
    """
    # Recompute CTR/CPC from daily totals for consistency with daily modeling granularity.
    temp = df[[date_col, "Ad_Spend", "Ad_CTR", "Ad_CPC", "Clicks", "Impressions"]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    result: dict[str, pd.Series] = {}
    spend = temp.groupby(date_col, as_index=True)["Ad_Spend"].sum().sort_index().astype(float)
    clicks = temp.groupby(date_col, as_index=True)["Clicks"].sum().sort_index().astype(float)
    impressions = temp.groupby(date_col, as_index=True)["Impressions"].sum().sort_index().astype(float)

    ctr = (clicks / impressions.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cpc = (spend / clicks.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    result["Ad_Spend"] = spend
    result["Ad_CTR"] = ctr.astype(float)
    result["Ad_CPC"] = cpc.astype(float)
    return result


def get_calendar_features(current_date: pd.Timestamp) -> dict[str, float]:
    """Create deterministic date/calendar features for one date."""
    # Calendar features are deterministic from date only, so they are safe for forecasting.
    us_holidays = holidays.US(years=current_date.year)
    current_date_only = current_date.date()
    is_holiday = float(current_date_only in us_holidays)
    days_to_holiday = min(
        [abs((h - current_date_only).days) for h in us_holidays.keys() if h.year == current_date.year] + [365]
    )
    days_until_month_end = float((current_date + pd.offsets.MonthEnd(0) - current_date).days)
    quarter_end = current_date + pd.offsets.QuarterEnd(0)
    days_until_quarter_end = float((quarter_end - current_date).days)
    shopping_events = {
        "is_black_friday": current_date.month == 11 and 24 <= current_date.day <= 30 and current_date.dayofweek == 4,
        "is_cyber_monday": current_date.month == 11 and 24 <= current_date.day <= 30 and current_date.dayofweek == 0,
        "is_new_year": current_date.month == 1 and 1 <= current_date.day <= 3,
        "is_thanksgiving_week": current_date.month == 11 and 20 <= current_date.day <= 30,
        "is_christmas_period": current_date.month == 12 and 15 <= current_date.day <= 31,
    }
    feats = {
        "dow": float(current_date.dayofweek),
        "day": float(current_date.day),
        "month": float(current_date.month),
        "quarter": float(current_date.quarter),
        "weekofyear": float(current_date.isocalendar().week),
        "is_weekend": float(current_date.dayofweek >= 5),
        "is_month_start": float(current_date.is_month_start),
        "is_month_end": float(current_date.is_month_end),
        "is_holiday": is_holiday,
        "days_to_holiday": float(min(days_to_holiday, 365)),
        "is_holiday_week": float(1 if is_holiday or days_to_holiday <= 3 else 0),
        "days_until_month_end": float(max(days_until_month_end, 0)),
        "days_until_quarter_end": float(max(days_until_quarter_end, 0)),
    }
    feats.update({k: float(v) for k, v in shopping_events.items()})
    return feats


def build_feature_row(
    current_date: pd.Timestamp,
    history: pd.Series,
    config: TargetConfig,
    ad_metrics: dict[str, pd.Series] | None = None,
) -> dict[str, float]:
    """Generate one supervised feature row from historical raw-scale target values.

    Notes:
    - Lag/rolling features are based on raw-scale history.
    - Missing future lag references fall back to the latest available value.
    """
    # Always anchor exogenous values to previous days to avoid same-day leakage.
    prev_day = current_date - pd.Timedelta(days=1)
    feats = get_calendar_features(current_date)
    feats["trend_idx"] = float((current_date - history.index.min()).days)
    feats["same_dow_mean_8"] = float(weekday_history_mean(history, current_date, lookback=8))
    for lag in config.lags:
        lag_date = current_date - pd.Timedelta(days=lag)
        val = history.get(lag_date, np.nan)
        if np.isnan(val) and not history.empty:
            # During recursive rollout, future lag dates may be missing early in the horizon.
            val = history.iloc[-1]  # Use last available value for future dates
        feats[f"lag_{lag}"] = float(val)
    for window in config.rolling_windows:
        window_start = prev_day - pd.Timedelta(days=window - 1)
        window_series = history.loc[window_start:prev_day]
        if len(window_series) == 0 and not history.empty:
            # For future dates, use the last available window
            last_date = history.index.max()
            window_start = last_date - pd.Timedelta(days=window - 1)
            window_series = history.loc[window_start:last_date]
        feats[f"roll_mean_{window}"] = float(window_series.mean()) if len(window_series) else float(history.mean())
        feats[f"roll_std_{window}"] = float(window_series.std()) if len(window_series) > 1 else 0.0
        feats[f"roll_min_{window}"] = float(window_series.min()) if len(window_series) else float(history.min())
        feats[f"roll_max_{window}"] = float(window_series.max()) if len(window_series) else float(history.max())
    if ad_metrics:
        for col, series in ad_metrics.items():
            val = series.get(prev_day, np.nan)
            if np.isnan(val) and not series.empty:
                # If an exogenous scenario is shorter than needed, carry the latest known value.
                val = series.iloc[-1]
            feats[f"{col}_prev_day"] = float(val)
            for lag in (7, 14):
                lag_date = current_date - pd.Timedelta(days=lag)
                val = series.get(lag_date, np.nan)
                if np.isnan(val) and not series.empty:
                    val = series.iloc[-1]
                feats[f"{col}_lag_{lag}"] = float(val)
    feats["lag_1_minus_lag_7"] = feats.get("lag_1", np.nan) - feats.get("lag_7", np.nan)
    feats["lag_7_minus_lag_28"] = feats.get("lag_7", np.nan) - feats.get("lag_28", np.nan)
    feats["roll_mean_7_minus_28"] = feats.get("roll_mean_7", np.nan) - feats.get("roll_mean_28", np.nan)
    return feats


def weekday_history_mean(history: pd.Series, current_date: pd.Timestamp, lookback: int = 8) -> float:
    """Return recent same-weekday mean as a seasonality proxy."""
    subset = history[history.index.dayofweek == current_date.dayofweek].tail(lookback)
    if subset.empty:
        return float(history.mean())  # Fallback to overall mean
    return float(subset.mean())


def build_feature_matrix(
    dates: pd.DatetimeIndex,
    history: pd.Series,
    config: TargetConfig,
    ad_metrics: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Build full supervised table for a date index and target history.

    Feature columns are raw-scale engineered features; target column is transformed.
    """
    rows: list[dict[str, float]] = []
    targets: list[float] = []
    for current_date in dates:
        if current_date <= history.index.min():
            # Skip rows without enough history to define lag-based predictors.
            continue
        rows.append(build_feature_row(current_date, history, config, ad_metrics))
        target_val = float(history.loc[current_date])
        # Apply transform to target
        transformed_target = transform_target(np.array([target_val]), config.transform)[0]
        targets.append(transformed_target)
    feature_df = pd.DataFrame(rows, index=dates)
    feature_df["target"] = targets
    return feature_df


def build_xgb_model(config: TargetConfig) -> XGBRegressor:
    """Instantiate an XGBRegressor with target-specific params."""
    return XGBRegressor(**config.model_params, objective="reg:squarederror", verbosity=0)


def compute_recency_weights(index: pd.DatetimeIndex, half_life_days: float | None) -> np.ndarray:
    """Return exponential recency weights aligned to a date index.

    If half_life_days is None or <= 0, returns uniform weights.
    """
    if half_life_days is None or half_life_days <= 0:
        return np.ones(len(index), dtype=float)

    idx = pd.to_datetime(index)
    max_date = idx.max()
    age_days = (max_date - idx).days.astype(float)
    # Each additional half-life halves the sample weight.
    weights = np.exp(-np.log(2.0) * (age_days / float(half_life_days)))
    return np.asarray(weights, dtype=float)


def recursive_forecast(
    model: XGBRegressor,
    history_raw: pd.Series,
    future_dates: pd.DatetimeIndex,
    config: TargetConfig,
    ad_metrics: dict[str, pd.Series] | None = None,
) -> np.ndarray:
    """Recursive multi-step rollout in transformed-output/raw-history mode.

    For each horizon step:
    1) Build features from raw history.
    2) Predict transformed target.
    3) Inverse-transform prediction to raw scale.
    4) Append raw prediction to history for next step.
    """
    # Keep autoregressive feature construction on the original scale.
    working_history_raw = history_raw.astype(float).copy()
    preds: list[float] = []

    for current_date in future_dates:
        row = build_feature_row(current_date, working_history_raw, config, ad_metrics)
        features = pd.DataFrame([row])
        # Model output remains in transformed scale; convert only for history update.
        pred_transformed = float(model.predict(features)[0])
        preds.append(pred_transformed)

        pred_raw = float(inverse_transform(np.array([pred_transformed]), config.transform)[0])
        working_history_raw = pd.concat(
            [working_history_raw, pd.Series([pred_raw], index=[current_date])]
        ).sort_index()

    return np.asarray(preds, dtype=float)

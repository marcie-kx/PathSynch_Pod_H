# Team XGBoost Delivery

This folder contains the minimum set of files needed to share the Pod H XGBoost pipeline.

## Included Files

### Code
- `rebuild_xgboost_pipeline/utils.py`
- `rebuild_xgboost_pipeline/build_features.py`
- `rebuild_xgboost_pipeline/backtest.py`
- `rebuild_xgboost_pipeline/forecast_30d.py`

### Docs
- `docs/XGBOOST_PIPELINE_SUMMARY.html`
- `docs/XGBOOST_VALIDATION_REPORT.md`

## Execution Order

```bash
/opt/homebrew/bin/python3.13 team_xgboost_delivery/rebuild_xgboost_pipeline/build_features.py
/opt/homebrew/bin/python3.13 team_xgboost_delivery/rebuild_xgboost_pipeline/backtest.py --target Revenue
/opt/homebrew/bin/python3.13 team_xgboost_delivery/rebuild_xgboost_pipeline/forecast_30d.py --target Revenue
```

## Validation Status

- Syntax check passed
- Feature generation passed
- Backtest passed
- 30-day forecast passed

## Final Choice

The single XGBoost Revenue model remains the final recommended approach, but the recursive evaluation is more conservative than the original optimistic backtest.

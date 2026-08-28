# Development Notes

## Milestone 1 — Data Pipeline

- Bug: yfinance `end` parameter is exclusive, not inclusive. It always excludes the last day requested. Fixed by shifting end date +1 day in `fetch_data`.
- Bug: `exchange_calendars` (via `pandas_market_calendars`) does not account for many Indonesian holidays. Verified against official BEI holiday announcements — 11/13 flagged gaps were confirmed legitimate holidays. Downgraded `trading_day_gaps` from hard error to warning.
- Found: BBCA.JK had suspended trading days (Volume=0, frozen OHLC) — added `detect_suspended_days` as a non-blocking warning check.

## Milestone 2 — Backtest Engine

- Bug: `return returns` initially placed inside the for-loop body (same indentation as if/else), causing the function to exit after the first iteration regardless of trade state. Fixed by dedenting to loop level.
- Edge case: a BUY signal on the last actionable day (i == end-2) opened a position that was never closed — mark-to-market logic was originally nested inside the else branch, only reachable if already in a position before that iteration. Fixed by moving mark-to-market check outside the loop entirely, run once after all iterations complete.
- Design decision: exit rule kept simple for v1 — position closes only on SELL signal or forced mark-to-market at data end. No stop-loss/invalidation logic yet (deferred to v2, noted as known limitation).
- Testing: caught floating-point precision bug (`0.1 != 0.10000000000000009`) via `pytest.approx` — direct `==` comparison on floats is unreliable.

## Milestone 3 - Feature Engineering

- Known limitation: `calculate_pyschological_level_proximity` has a division-by-zero risk when price rounds to a psychological level of 0 (ratio=0). Not handled — BBCA.JK's price range (thousands of IDR) makes this practically unreachable within project scope. Would need guarding if generalized to other tickers with very low share prices 

## Milestone 4 — Model Training & Threshold Experiment

### 4.1 Model Training

- Built `TimeSeriesSplit`-based GridSearchCV with a custom scorer (`f1_trading_signals`) that excludes the HOLD class — default `f1_macro` was found to mask poor BUY/SELL performance behind easy HOLD predictions (0.70 vs 0.58 in a quick sanity test).
- Label distribution with threshold=0.02: HOLD=871, BUY=95, SELL=77 (~38 actionable signals/year — meets the Success Metric minimum).
- Cross-validated trading F1 score: 0.153 — well below in-sample precision (0.24-0.28), which itself is below the 50% Success Metric target.
- Reflection: 5 technical features may be insufficient to predict >2% daily moves in a highly liquid stock like BBCA.JK. This is consistent with efficient market behavior, not necessarily a methodology flaw — the walk-forward validation and leakage-safe scorer are working correctly; they're revealing a genuine signal-strength limitation.
- Next steps to explore: lower threshold, additional/interaction features, or accept modest performance and document honestly in the final writeup.

### 4.2 Threshold Experiment

- Conducted a small systematic threshold search (`0.02`, `0.015`, `0.01`) to evaluate how the labeling threshold affects BUY/SELL prediction performance, rather than selecting a value arbitrarily.
- Lower thresholds consistently improved cross-validated trading F1, suggesting that the current feature set is better at predicting smaller price movements than larger (>2%) moves.
- However, higher F1 does not necessarily imply higher economic value: lower thresholds generate more marginal signals with smaller expected price movements, making transaction costs and slippage more significant.   
- Selected `threshold=0.01` as the baseline for Milestone 5 — a reasonable balance between predictive performance and trading significance, without continuing to optimize solely for F1.   
- Reflection: this experiment serves as a small hyperparameter search rather than an attempt to overtune the model to a single ML metric. A predictive model does not necessarily imply a profitable trading strategy; higher F1 means the model is better at predicting the defined labels, while profitability must be validated through out-of-sample trading performance after transaction costs.

## Milestone 5 - Confluence / Signal Generation
- Built a function to confluence between prediction from the model and `moving_average_trend` as a veto. Avoided overly complex and sophisticated method by not  choosing or combining all methods or technique such as: `weight_scoring`, `confidence_level`, etc.
- Chose to not use square root of 252 as `trades_per_year` to avoid exaggerating the annualized `sharpe_ratio`.
Did full pipeline test: from `load_and_validate_data` to `metrics`, turns out the `moving_average_trend` veto doesn't bring any positive impact (at least to this experiment). 
- Findings: 
    - With veto -> 10 trades, negative expectancy(approximately -0.0168)
    - Without veto -> 90 trades, positive expectancy(approximately 0.0014-0.0021)
    - Annualized `sharpe_ratio` is around ~0.17–0.25, with buy-and-hold is around ~0.13 both considered low in absolute.
- Chose to not use veto as baseline, not because it proven to be bad, but because the samples are too little.
- Limitations: number of trades and metrics changes with every run, although the `random_state` is already set to equal 42. Small sample size, only find 90 trades in more than 4 years is not enough to support a strong statistical claim. 
- The feature limitation identified in Milestone 4 still persists here, the features are just simply not enough to describe, reflect, and predict the blue-chip stocks that tend to have a dynamic price movements.

## Milestone 6 - End-to-End Integration
- Built:
    - A `main.py` module to orchestrate the entire ML and run the backtesting pipeline without any manual intervention.
    - A `persistence.py` module as a helper for `main.py` to interact with to store the model or fetch it.
- Chose to make `/outputs` as a directory that store the existing model, so that no training needed when already did it once.
- Added some configs to make every parameter is not hardcoded; note: all the parameters that are included in config are limited, only when it needs to be experimented with or a result of an experiment(such as: `threshold`).
- Did some refactoring on `train_model` and `backtest_signals` functions due to config adjustments, so that it can accept config input as parameters.

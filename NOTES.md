# Development Notes

## Milestone 1 - Data Pipeline

- Bug: check_duplicates initially used df.duplicated() (row-value based), which incorrectly flagged suspended-trading days with identical frozen OHLCV values as duplicates. Fixed by switching to df.index.duplicated(), since the actual definition of a duplicate we needed was the same trading date appearing more than once, not identical row values. 

- Bug: yfinance `end` parameter is exclusive. It always excludes the last day requested. Fixed by shifting end date +1 day in `fetch_data`.

- Bug: `exchange_calendars` (via `pandas_market_calendars`) does not account for many Indonesian holidays. Verified against official BEI holiday announcements with 11 of 13 flagged gaps confirmed as legitimate holidays. 

- Decision: Downgraded `trading_day_gaps` from hard error to warning.

- Finding: BBCA.JK had suspended trading days (Volume=0, frozen OHLC). Added `detect_suspended_days` as a non-blocking warning check.

## Milestone 2 - Backtest Engine

- Decision: Built and unit-tested the backtest engine using dummy/rule-based signals before connecting it to the ML model. This made it easier to determine whether any surprising results came from the model/data or from a bug in the backtesting logic.

- Bug: `return returns` initially placed inside the for-loop body (same indentation as if/else), causing the function to exit after the first iteration regardless of trade state. Fixed by dedenting to loop level.

- Edge case: a BUY signal on the last actionable day (i == end-2) opened a position that was never closed, mark-to-market logic was originally nested inside the else branch, only reachable if already in a position before that iteration. Fixed by moving mark-to-market check outside the loop entirely, run once after all iterations complete.

- Design decision: exit rule kept simple, position closes only on SELL signal or forced mark-to-market at data end. No stop-loss/invalidation logic yet.

- Bug: On testing, caught floating-point precision issue. Direct `==` comparison on floats is unreliable  (`0.1 != 0.10000000000000009`). Fixed it with `pytest.approx`.

## Milestone 3 - Feature Engineering
- Decision: Split each feature into a raw-value calculation and a separate threshold/flag function such as calculate_volume_zscore + flag_volume_anomalies instead of combining both in one function. This made the functions easier to test and kept each function focused on a single responsibility.

- Limitation: `calculate_psychological_level_proximity` has a division-by-zero risk when price rounds to a psychological level of 0 (ratio=0). Not fixed or handled, BBCA.JK's price range (thousands of IDR) makes this practically unreachable within project scope. Would need guarding if generalized to other tickers with very low share prices 

## Milestone 4 - Model Training & Threshold Experiment

### 4.1 Model Training

- Built `TimeSeriesSplit`-based GridSearchCV with a custom scorer (`f1_trading_signals`) that excludes the HOLD class, because default `f1_macro` was found to mask poor BUY/SELL performance behind easy HOLD predictions (0.70 vs 0.58 in a quick sanity test).

- Finding: Label distribution with threshold=0.02: HOLD=871, BUY=95, SELL=77 (~38 actionable signals/year which meets the Success Metric minimum).

- Finding: Cross-validated trading F1 score: 0.153, well below in-sample precision (0.24-0.28), which itself is below the 50% Success Metric target.

- Reflection: 5 technical features may be insufficient to predict >2% daily moves in a highly liquid stock like BBCA.JK. This is consistent with efficient market behavior, not necessarily a methodology flaw, the walk-forward validation and leakage-safe scorer are working correctly, they're just revealing a genuine signal-strength limitation.

- Next steps to explore: lower threshold, additional/interaction features, or accept modest performance and document honestly in the final writeup.

### 4.2 Threshold Experiment

- Conducted a small systematic threshold search (`0.02`, `0.015`, `0.01`) to evaluate how the labeling threshold affects BUY/SELL prediction performance, rather than selecting a value arbitrarily.

- Finding: Lower thresholds consistently improved cross-validated trading F1, suggesting that the current feature set is better at predicting smaller price movements than larger (>2%) moves.

- Finding: However, lower thresholds generate more marginal signals with smaller expected price movements, making transaction costs and slippage more significant.   

- Decision: Selected `threshold=0.01` as the baseline for Milestone 5, quite a reasonable balance between predictive performance and trading significance, without continuing to optimize solely for F1.   

- Reflection: this experiment serves as a small-scale hyperparameter search rather than an attempt to overtune the model to a single ML metric. A predictive model does not necessarily imply a profitable trading strategy. Higher F1 just means the model is better at predicting the defined labels, while profitability must be validated through performance from many sample of trades.

## Milestone 5 - Confluence / Signal Generation

- Built: `src/signals/confluence.py` to make a confluence between prediction from the model and `moving_average_trend` as a veto. 

- Finding: Did full pipeline test from `load_and_validate_data` to `metrics`. It turns out the `moving_average_trend` veto doesn't bring any positive impact (at least to this experiment). 

- Finding: With veto -> 10 trades, negative expectancy (approximately -0.0168)

- Finding Without veto -> 90 trades, positive expectancy (approximately 0.0014-0.0021)

- Finding: Annualized `sharpe_ratio` is around ~0.17–0.25, with buy-and-hold is around ~0.13 both considered low in absolute.

- Decision: Avoided overly complex and sophisticated method by not  choosing or combining all methods or technique such as: `weight_scoring`, `confidence_level`, etc.

- Decision: Chose to not use square root of 252 as `trades_per_year` to avoid exaggerating the annualized `sharpe_ratio`.

- Decision: Chose to not use veto as baseline, not because it proved to be bad, but because the sample size was too small.

- Limitation: The number of trades and metrics changes with every run, although the `random_state` is already set to equal 42. Small sample size, only 90 trades were found in more than 4 years is not enough to support a strong statistical claim. 

- Limitation: The feature limitation identified in Milestone 4 still persists here, the features are just simply not enough to describe, reflect, and predict the blue-chip stocks that tend to have a dynamic price movements.

## Milestone 6 - End-to-End Integration

- Built: `main.py` module to orchestrate the entire ML and run the backtesting pipeline without any manual intervention.

- Built: `src/models/persistence.py` module as a helper for `main.py` to interact with to store the model or fetch it.

- Bug: The explicit `.dropna()` step from confluence.py was missed when porting the logic to main.py, which caused NaN values to reappear in the signal column. It happened to be harmless because NaN == 1 evaluates to False, but it was fixed to avoid relying on this behavior.

- Bug: accidentally use hardcoded values in `train_model` instead of the function parameters. Fixed it by using provided parameters.

- Decision: Chose to make `/outputs` as a directory that store the existing model, so that no training needed when already did it once.

- Design decision: Added some configs to make every parameter is not hardcoded. Note: all the parameters that are included in config are limited, only when it needs to be experimented with or a result of an experiment(such as: `threshold`).

- Design decision: Did some refactoring on `train_model` and `backtest_signals` functions due to config adjustments, so that it can accept config input as parameters.

## Milestone 7 - Streamlit Dashboard

- Built: `app/dashboard.py` with 4 components: overview/key metrics, price chart with BUY/SELL signal markers (+ date range filter), backtest performance (expectancy, Sharpe vs buy-and-hold), and feature importance.

- Built: `src/backtest/comparison.py` to calculate and compare strategy returns against a buy-and-hold DCA benchmark, refactored from app/dashboard.py to separate backtest comparison logic from the Streamlit presentation layer.

- Bug: Forgot to use `.pct_change().dropna()` on `buy_hold_returns` and made the calculation for `buy_hold_sharpe` with actual price not with returns day-to-day.

- Bug: filtering `predictions` for BUY/SELL signals initially attempted `df.loc[predictions == 1]` directly, which failed with an `IndexingError`. Fixed it by matching the index with the actual `dataframe` first.

- Bug: component didn't appear in order due to differences between Python execution order and Streamlit's render order.

- Decision: Chose not to call `main.py` to runs the full training pipeline , while the dashboard only needs to load an already-trained model and generate predictions.

- Decision: Used `@st.cache_resource` and `@st.cache_data` to caching the model and the data, avoids expensive recomputation on every user interaction.

- Decision:  Backtest performance metrics are computed from the full dataset, independent of the chart's date range filter, due to limitations: small trades samples.

- Decision:  Feature importance uses built-in `feature_importances_` attribute (no extra computation needed).

- Decision: Decided to remove `app/api.py` from project scope. The Streamlit dashboard already sufficient as it is and keeping FastAPI would just add another complexity without providing any additional value (at least at this stage). 

- Decision: Replaced Matplotlib with Plotly for interactive dashboard visualizations and added plotly to requirements.txt to ensure the new visualization dependency is included for reproducibility.

- Decision: Added `backtest.initial_capital` to `config.yaml` to make the initial backtest capital configurable instead of hard-coded in the dashboard.

- Design decision (UX): Added a date range picker to avoid an overly crowded chart, with  approximately 590 raw model signals across 4.5 years, an unfiltered chart was too dense to read.

- Design decision (UI): Did some polish by adding some caption to each section and using `color="green"`/`color="red"` for BUY/SELL markers to match standard financial charting conventions.

- Design decision (UI): Moved the date range picker to the sidebar to keep the main dashboard clear and prevent filtering controls from distracting from the primary visualizations.

- Design decision (UI): Added a research disclaimer explaining that backtest results are based on historical data and require further validation before live deployment.

## Milestone 8 - Polish

- Decision: Standardized `NOTES.md` formatting with consistent labels (`Bug:`, `Decision:`, `Finding:`, `Limitation:`, `Reflection:`) across all milestones to improve readability and consistency.

- Built: `README.md` as the final project documentation, covering the project overview, business problem, architecture, dashboard, results, limitations, and setup instructions.

- Decision: Added an `assets/` directory to store dashboard screenshots used in `README.md`. Added a dedicated `Dashboard` section directly after `Overview` to provide a visual preview of the project before the more detailed technical sections.

- Decision: Documented run-to-run non-determinism as a known limitation, including the observed performance range across multiple runs, rather than root-causing it within the remaining project time.

- Decision: Reported the final results as an observed range across multiple runs instead of presenting a single run. This decision was made after observing noticeable variation in the results during final testing, making the range a more honest representation of the model's performance.

- Reflection: A single run can give a misleading impression of model performance when the results vary between runs. Reporting the observed range gives a more honest picture of the system's behavior during development.

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
- Known limitation: `calculate_pyschological_level_proximity` has a division-by-zero risk when price rounds to a psychological level of 0 (ratio=0). Not handled — BBCA.JK's price range (thousands of IDR) makes this practically unreachable within project scope. Would need guarding if generalized to other tickers with very low share prices (e.g. penny stocks).
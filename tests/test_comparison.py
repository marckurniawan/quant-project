import pandas as pd
import pytest

from src.backtest.comparison import calculate_strategy_vs_buy_and_hold


def test_calculate_strategy_vs_buy_and_hold():
    dates = pd.date_range("2022-01-03", periods=5, freq="D")

    df = pd.DataFrame(
        {"Close": [100, 110, 120, 130, 140]},
        index=dates,
    )

    strategy_returns = pd.Series([0.10, -0.05, 0.20])

    investment_dates = [
        pd.Timestamp("2022-01-03"),
        pd.Timestamp("2022-01-05")
    ]

    result = calculate_strategy_vs_buy_and_hold(
        df=df,
        strategy_returns=strategy_returns,
        initial_capital=100.0,
        investment_dates=investment_dates,
    )

    # Strategy:
    # 100 * 1.10 * 0.95 * 1.20 = 125.40
    assert result["strategy_final_value"] == pytest.approx(125.40)
    assert result["strategy_profit"] == pytest.approx(25.40)
    assert result["strategy_return"] == pytest.approx(0.254)

    # $50 invested at 100 -> $70
    # $50 invested at 120 -> $58.3333
    # Final value = $128.3333
    assert result["periodic_final_value"] == pytest.approx(128.3333333)
    assert result["periodic_profit"] == pytest.approx(28.3333333)
    assert result["periodic_return"] == pytest.approx(0.283333333)

    assert result["total_dca"] == 2
    assert result["investment_amount"] == pytest.approx(50.0)


def test_buy_and_hold_uses_next_available_trading_day():
    dates = pd.to_datetime([
        "2022-01-03",
        "2022-01-05",
        "2022-01-06"
    ])

    df = pd.DataFrame(
        {"Close": [100, 110, 120]},
        index=dates
    )

    strategy_returns = pd.Series([0.10])

    # 2022-01-04 is not a trading day.
    # Function should use 2022-01-05 instead.
    investment_dates = [
        pd.Timestamp("2022-01-04"),
    ]

    result = calculate_strategy_vs_buy_and_hold(
        df=df,
        strategy_returns=strategy_returns,
        initial_capital=100.0,
        investment_dates=investment_dates,
    )

    # $100 invested at 110 and sold at 120
    expected_value = 100 * (120 / 110)

    assert result["periodic_final_value"] == pytest.approx(expected_value)


def test_buy_and_hold_excludes_dates_after_dataset():
    dates = pd.date_range("2022-01-03", periods=3, freq="D")

    df = pd.DataFrame(
        {"Close": [100, 110, 120]},
        index=dates,
    )

    strategy_returns = pd.Series([0.10])

    investment_dates = [
        pd.Timestamp("2022-01-03"),
        pd.Timestamp("2022-01-10")]

    result = calculate_strategy_vs_buy_and_hold(
        df=df,
        strategy_returns=strategy_returns,
        initial_capital=100.0,
        investment_dates=investment_dates,
    )

    assert result["total_dca"] == 1
    assert result["investment_amount"] == pytest.approx(100.0)

    # Only 2022-01-03 should be used
    # 100 -> 120
    assert result["periodic_final_value"] == pytest.approx(120.0)


def test_dca_raises_error_when_no_valid_dates():
    dates = pd.date_range("2022-01-03", periods=3, freq="D")

    df = pd.DataFrame(
        {"Close": [100, 110, 120]},
        index=dates,
    )

    strategy_returns = pd.Series([0.10])

    investment_dates = [pd.Timestamp("2022-01-10")]

    with pytest.raises(ValueError, match="No investment dates"):
        calculate_strategy_vs_buy_and_hold(
            df=df,
            strategy_returns=strategy_returns,
            initial_capital=100.0,
            investment_dates=investment_dates,
        )


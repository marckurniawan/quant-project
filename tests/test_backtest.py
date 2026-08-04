import pandas as pd
import pytest

from src.backtest.engine import close_position
from src.backtest.engine import backtest_signals    

def test_close_position_returns_profit():
    result = close_position(
        entry_price=100,
        exit_price=110,
        buy_fee=0,
        sell_fee=0,
    )

    assert result == pytest.approx(0.10)


def test_close_position_returns_loss():
    result = close_position(
        entry_price=100,
        exit_price=80,
        buy_fee=0,
        sell_fee=0,
    )

    assert result == pytest.approx(-.20)


def test_close_position_raises_when_entry_price_is_zero():
    with pytest.raises(
        ValueError,
        match="entry_price must be positive",
    ):
        close_position(
            entry_price=0,
            exit_price=100,
        )


def test_backtest_returns_empty_when_no_signals():
    df = pd.DataFrame(
        {
            "Close": [100, 101, 102, 103],
            "signal": [0, 0, 0, 0],
        }
    )

    assert backtest_signals(df) == []


def test_backtest_force_closes_open_position():
    df = pd.DataFrame(
        {
            "Close": [100, 105, 110, 120],
            "signal": [1, 0, 0, 0],
        }
    )

    returns = backtest_signals(df)

    expected = close_position(
        entry_price=105,
        exit_price=120,
    )

    assert returns == [expected]

def test_backtest_executes_multiple_trades():
    df = pd.DataFrame(
        {
            "Close": [100, 105, 110, 115, 120, 125, 130],
            "signal": [1, 0, -1, 1, 0, -1, 0],
        }
    )

    returns = backtest_signals(df)

    trade_1 = close_position(
        entry_price=105,
        exit_price=115,
    )

    trade_2 = close_position(
        entry_price=120,
        exit_price=130,
    )

    assert returns == [trade_1, trade_2]
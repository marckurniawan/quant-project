import pandas as pd
import pytest
from sklearn.metrics import f1_score

from  src.models.training import f1_trading_signals


def test_f1_perfect_condition():
    y_true = pd.Series([1, 1, -1, 0, 0, -1, -1])

    y_pred = pd.Series([1, 1, -1, 0, 0, -1, -1])

    assert f1_trading_signals(y_true=y_true, y_pred=y_pred) == pytest.approx(1)


def test_f1_trading_signals_counts_hold_as_false_positive():
    y_true = pd.Series([1, 1, -1, 0, 0, -1, -1])

    y_pred = pd.Series([1, 1, -1, 1, 1, -1, -1])

    # Calculation
    # BUY: TP=2, FP=2, FN=0 → F1=0.6667
    # SELL: TP=3, FP=0, FN=0 → F1=1.0
    # Macro F1 = (0.6667 + 1.0) / 2 = 0.8333
    assert f1_trading_signals(y_true=y_true, y_pred=y_pred) == pytest.approx(0.8333333)
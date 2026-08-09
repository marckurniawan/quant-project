import pandas as pd

from src.features.engineering import calculate_ma_trend

def test_calculate_ma_trend_when_fast_above_slow():
    df= pd.DataFrame({"Close":[10, 11, 11, 17, 18]})

    assert calculate_ma_trend(df, fast=2, slow=3).iloc[-1] == 1

def test_calculate_ma_trend_whenslow_above_fast():
    df= pd.DataFrame({"Close":[15, 17, 18, 17, 14]})

    assert calculate_ma_trend(df, fast=2, slow=3).iloc[-1] == -1


def test_calculate_ma_trend_when_fast_and_slow_are_equal():
    df= pd.DataFrame({"Close":[15, 17, 18, 21, 15]})

    assert calculate_ma_trend(df, fast=2, slow=3).iloc[-1] == 0
import pandas as pd
import pytest

from src.data.data_loader import check_duplicates
from src.data.data_loader import detect_suspended_days


def test_check_duplicates_detects_duplicate_index():
    df = pd.DataFrame(
        {"Price": [100, 200, 300]},
        index=["2022-10-22", "2022-10-22", "2022-10-23"],
        
    )

    assert check_duplicates(df) == 1



def test_check_duplicates_returns_zero_when_unique():
    df = pd.DataFrame(
        {"Price": [100, 200, 300]},
        index=["2022-10-22", "2022-10-23", "2022-10-24"],
    )

    assert check_duplicates(df) == 0

def test_detect_suspended_days_when_no_suspension():
    df = pd.DataFrame(
        {"Price" :[100, 200, 300],
         "Volume": [10, 20, 30]},
         index=pd.to_datetime(["2022-10-22", "2022-10-23", "2022-10-24"])
    )
    assert detect_suspended_days(df) == []

def test_detect_suspended_days_returns_when_suspension_exists():
    df = pd.DataFrame(
            {"Price" :[100, 200, 300],
             "Volume": [0, 20, 0]},
             index=pd.to_datetime(["2022-10-22", "2022-10-23", "2022-10-24"])
        )
    assert detect_suspended_days(df) == list(pd.to_datetime(["2022-10-22", "2022-10-24"]))
import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def fetch_data(ticker:str, start:str, end:str)->pd.DataFrame:
    """Download OHLCV from Yahoo Finance"""
    df = yf.download(ticker , start=start, end=end, auto_adjust=True, progress=False)
    df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        logging.error(f"No data found for ticker '{ticker}'.")
        raise ValueError(f"No data found for ticker '{ticker}'") 
    return df


def check_missing_values(df:pd.DataFrame)->int:
    """Count total missing values."""
    return int(df.isna().sum().sum())


def check_duplicates(df:pd.DataFrame)->int:
    """Count total duplicate rows."""
    return int(df.index.duplicated().sum())


def get_expected_trading_days(start:str, end:str)->pd.DatetimeIndex:
    """Get an official trading days from exchange calendar."""
    calendar = mcal.get_calendar("XIDX")

    schedule = calendar.schedule(start_date=start, end_date=end)

    return schedule.index.normalize()


def trading_day_gaps(df:pd.DataFrame, expected_dates:pd.DatetimeIndex)->list[pd.Timestamp]:
    """Detect missing trading days."""
    actual_dates = pd.DatetimeIndex(df.index).normalize()

    gaps = expected_dates.difference(actual_dates)

    return list(gaps)


def detect_suspended_days(df: pd.DataFrame,) -> list[pd.Timestamp]:
    """Detect trading days with zero volume."""
    suspended_dates = df.index[df["Volume"] == 0]

    return list(suspended_dates)


def load_and_validate_data(ticker:str, start:str, end:str)->pd.DataFrame:
    df = fetch_data(ticker, start, end)
    logging.info(f"Successfully downloaded {len(df)} rows for {ticker}.")

    missing_values = check_missing_values(df)
    if missing_values > 0:
        logging.error(f"Validation failed: {missing_values} missing values.")
        raise ValueError (f"Pipeline cannot be run there's {missing_values} missing value(s).")
    logging.info(f"Validation passed: no missing values found.")
    
    duplicates = check_duplicates(df)
    if duplicates > 0:
        logging.error(f"Validation failed: {duplicates} duplicates.")
        raise ValueError (f"Pipeline cannot be run there's {duplicates} duplicates(s).")
    logging.info(f"Validation passed: no duplicates found.")

    expected_dates = get_expected_trading_days(start, end)
    gaps = trading_day_gaps(df, expected_dates)
    if gaps:
        logging.error(f"Validation failed: {len(gaps)} gaps.")
        raise ValueError (f"Pipeline cannot be run there's {len(gaps)} gap(s).")  
    logging.info(f"Validation passed: {len(gaps)} gaps.")

    suspended_days = detect_suspended_days(df)
    if suspended_days:
        logging.warning(
            f"Detected {len(suspended_days)} suspended day(s). "
            f"First dates: {suspended_days[:5]}"
        )
    else:
        logging.info("Validation passed: 0 suspended days.")    

    return df  
if __name__ == "__main__":
    
    df = load_and_validate_data("BBCA.JK", "2022-01-01", "2026-07-27")
    
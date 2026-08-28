import pandas as pd
import numpy as np

def calculate_volatility(df: pd.DataFrame, window: int = 20) ->  pd.Series:
    returns = df["Close"].pct_change()
    volatility = returns.rolling(window=window) .std()
    return volatility

def calculate_roc(df: pd.DataFrame, window: int = 20) ->  pd.Series:
    return df["Close"].pct_change(periods=window)  * 100


def calculate_ma_trend(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    ma_fast = df["Close"].rolling(window=fast).mean()
    ma_slow = df["Close"].rolling(window=slow).mean()
    diff = ma_fast - ma_slow
    return np.sign(diff)

def calculate_volume_zscore(df: pd.DataFrame, window: int = 20) ->  pd.Series:
    rolling_mean = df["Volume"].rolling(window=window) .mean()
    rolling_std = df["Volume"].rolling(window=window) .std()
    z_score =  (df["Volume"] - rolling_mean) / rolling_std

    return z_score


def flag_volume_anomalies(z_score: pd.Series, threshold: float = 2) ->  pd.Series:
    
    return z_score.abs() > threshold


def calculate_psychological_level_proximity(df: pd.DataFrame, interval: float = 500) ->  pd.Series:
    ratio = (df["Close"]/interval).round()
    psychological_level = ratio *  interval

    distance = np.abs(psychological_level - df["Close"])/psychological_level

    return distance


def flag_near_psych_level(distance: pd.Series, tolerance: float = 0.02) -> pd.Series:

    return distance <= tolerance


def generate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_volatility = calculate_volatility(df)
    rate_of_change = calculate_roc(df)
    moving_average_trend = calculate_ma_trend(df)

    z_score = calculate_volume_zscore(df)
    volume_anomalies = flag_volume_anomalies(z_score)

    psychological_level_proximity = calculate_psychological_level_proximity(df)
    near_psychological_level = flag_near_psych_level(psychological_level_proximity)

    features = pd.DataFrame({
        "volatility": rolling_volatility,
        "rate_of_change": rate_of_change,
        "moving_average_trend": moving_average_trend,
        "volume_anomaly": volume_anomalies,
        "near_psychological_level": near_psychological_level
    })

    return features




if __name__ == "__main__":
    # Dummy data for testing 
    df = pd.DataFrame({
        "Close": [100, 101, 102, 103, 104] * 10,  
        "Volume": [1000, 1100, 900, 1200, 950] * 10,
    })    
    
    features = generate_all_features(df)

    print(features)
   
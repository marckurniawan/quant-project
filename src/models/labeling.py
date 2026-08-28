import pandas as pd

def create_labels(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """Create labels based on next-day returns."""
    df = df.copy() 

    future_return = df["Close"].shift(-1)/ df["Close"] - 1 

    df["target"] = 0 

    df.loc[future_return > threshold, "target" ] = 1 
    df.loc[future_return < -threshold, "target" ] = -1 

    df["future_return"] = future_return 
    df = df.dropna(subset=["future_return"]) 


    return df
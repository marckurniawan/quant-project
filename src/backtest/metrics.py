import pandas as pd
import numpy as np

def calculate_expectancy(signals: pd.Series) -> float:
    wins = signals[signals > 0]
    losses = signals[signals < 0]

    win_rate  = len(wins)/ len(signals)
    loss_rate = len(losses)/ len(signals)

    avg_win = wins.mean()
    avg_losses = losses.mean()


    return (win_rate * avg_win) + (loss_rate * avg_losses)

def calculate_sharpe_ratio(returns: pd.Series, trades_per_year: int):
    # Multiply Sharpe by the square root of trades per year to annualize it
    return returns.mean() / returns.std() * np.sqrt(trades_per_year)
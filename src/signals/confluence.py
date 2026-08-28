import pandas as pd

from src.data.data_loader import load_and_validate_data
from src.models.training import prepare_training_data, train_model

from src.backtest.engine import backtest_signals
from src.backtest.metrics import calculate_expectancy
from src.backtest.metrics import calculate_sharpe_ratio

# Used to veto conflicting signals
def generate_signals(predictions: pd.Series,ma_trend: pd.Series) -> pd.Series:

    signals = predictions.copy()

    # MA trend acts as a veto, blocking ML signals that conflict with the broader trend.
    buy_conflict = (predictions == 1) & (ma_trend == -1)
    sell_conflict = (predictions == -1) & (ma_trend == 1)

    signals[buy_conflict] = 0
    signals[sell_conflict] = 0

    return signals


if __name__ == "__main__":
    df = load_and_validate_data("BBCA.JK", "2022-01-01", "2026-07-27")
    dataset = prepare_training_data(df, threshold=0.01)

    X = dataset.drop(columns=["target"])
    y = dataset["target"]

    model, validation_score, best_params = train_model(X, y)
    predictions = pd.Series(model.predict(X),index=X.index,name="prediction")

    backtest_df = pd.DataFrame({
        "signal": predictions,
        "Close": df["Close"]  # real dataframe
    })
    backtest_df = backtest_df.dropna()

    returns = backtest_signals(backtest_df)
    returns = pd.Series(returns)

    # Use 365.25 instead of 365 to account for leap years
    years = (df.index.max() - df.index.min()).days / 365.25
    total_trades = len(returns)
    trades_per_year = total_trades / years

    print("Number of trades:", total_trades)
    print("Total trades returns (sample):", returns[:-1])

    print("Expectancy:", calculate_expectancy(returns))
    print("Annualized Sharpe:", calculate_sharpe_ratio(returns,trades_per_year=trades_per_year))

    # Comparing to buy-hold strategy
    daily_returns = df["Close"].pct_change()
    print("Buy-Hold Annualized Sharpe:", calculate_sharpe_ratio(daily_returns, trades_per_year=252))
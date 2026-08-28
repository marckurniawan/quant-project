import pandas as pd
import yaml
from src.data.data_loader import load_and_validate_data
from src.models.training import prepare_training_data, train_model
from src.backtest.engine import backtest_signals
from src.backtest.metrics import calculate_expectancy, calculate_sharpe_ratio
from src.models.persistence import save_model

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    """An orchestrator for running the entire pipeline"""
    config = load_config()

    # Load and cheking the data
    df = load_and_validate_data(config["ticker"], start=config["data"]["start_date"], end=config["data"]["end_date"])

    # Prepare the data before train stage
    dataset = prepare_training_data(df, config["model"]["threshold"])
    
    X = dataset.drop(columns=["target"])
    y = dataset["target"]
    # Train the data using TimeSeriesSplit
    model, validation_score, best_params = train_model(X, y, random_state=config["model"]["random_state"], n_estimators=config["model"]["param_grid"]["n_estimators"], max_depth=config["model"]["param_grid"]["max_depth"], n_splits=config["model"]["n_splits"])

    # Save model
    save_model(model, "outputs/model.pkl    ")
    predictions = pd.Series(model.predict(X), index=X.index, name="prediction")

    backtest_data = (pd.DataFrame({"signal" : predictions,
                                   "Close"  : df["Close"]}))
    backtest_data = backtest_data.dropna()
    returns = backtest_signals(backtest_data, buy_fee=config["backtest"]["buy_fee"], sell_fee=config["backtest"]["sell_fee"])
    returns = pd.Series(returns)

    # Divided by 365.25 to convert to years
    duration = (df.index.max() - df.index.min()).days / 365.25
    total_trades = len(returns)
    trades_per_year = total_trades/duration

    # Calculate metrics
    expectancy = calculate_expectancy(returns)
    annualized_sharpe_ratio = calculate_sharpe_ratio(returns, trades_per_year=trades_per_year)

    # Print the result
    print("Number of trades:", total_trades)
    print("Total trades returns (sample):", returns[:5])
    print("Expectancy:", expectancy)
    print("Annualized Sharpe:", annualized_sharpe_ratio)


if __name__ == "__main__":
    main()
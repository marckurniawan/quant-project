import pandas as pd


def calculate_strategy_vs_buy_and_hold(
    df: pd.DataFrame, strategy_returns: pd.Series, initial_capital: float, investment_dates: list[pd.Timestamp],) -> dict:
    """
    Compare a compounded trading strategy with periodic buy-and-hold investment.
    """

    # Keep only investment dates covered by the dataset
    valid_investment_dates = [
        date for date in investment_dates
        if date <= df.index.max()
    ]

    total_dca = len(valid_investment_dates)

    if total_dca == 0:
        raise ValueError("No investment dates are within the dataset.")

    investment_amount = initial_capital / total_dca

    # Strategy: compound all trade returns
    strategy_equity = (
        initial_capital * (1 + strategy_returns).cumprod()
    )

    strategy_final_value = strategy_equity.iloc[-1]
    strategy_profit = strategy_final_value - initial_capital
    strategy_return = strategy_final_value / initial_capital - 1

    # DCA: invest an equal amount on each investment date
    periodic_final_value = 0.0

    for date in valid_investment_dates:

        # Find the first available trading day on or after the investment date
        available_dates = df.index[df.index >= date]

        if len(available_dates) == 0:
            continue

        entry_date = available_dates[0]
        entry_price = df.loc[entry_date, "Close"]

        # Hold the investment until the end of the dataset
        exit_price = df["Close"].iloc[-1]

        investment_return = exit_price / entry_price - 1

        periodic_final_value += (
            investment_amount * (1 + investment_return)
        )

    periodic_profit = periodic_final_value - initial_capital
    periodic_return = periodic_profit / initial_capital

    return {
        "strategy_final_value": strategy_final_value,
        "strategy_profit": strategy_profit,
        "strategy_return": strategy_return,
        "periodic_final_value": periodic_final_value,
        "periodic_profit": periodic_profit,
        "periodic_return": periodic_return,
        "total_dca": total_dca,
        "investment_amount": investment_amount,
    }

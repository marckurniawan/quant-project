import pandas as pd 

def close_position(entry_price: float, exit_price: float, buy_fee: float = 0.0015,    sell_fee: float = 0.0025) -> float:
    """ Calculate net return after transaction costs."""

    if entry_price <= 0:
        raise ValueError("entry_price must be positive")

    cost_basis = entry_price * (1 + buy_fee)
    net_proceeds = exit_price * (1 - sell_fee)

    return net_proceeds / cost_basis - 1


def backtest_signals(df: pd.DataFrame, signal_col: str = "signal", price_col: str = "Close") -> list[float]:
    """ Execute trades using T+1 execution. """

    returns = []
    position = False
    entry_price = None

    end = len(df)

    for i in range(end - 1):
        signal = df[signal_col].iloc[i]

        if not position:

            if signal == 1:
                entry_price = float(df[price_col].iloc[i + 1])
                position = True

        else:

            if signal == -1:
                exit_price = float(df[price_col].iloc[i + 1])

                returns.append(
                    close_position(
                        entry_price,
                        exit_price,
                    )
                )

                position = False
        
# Force close any remaining position 
    if position:
        exit_price = float(df[price_col].iloc[-1])

        returns.append(
            close_position(
                entry_price,
                exit_price,
            )
        )
            

    return returns
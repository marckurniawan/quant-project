

def generate_signals(predictions: pd.Series,ma_trend: pd.Series) -> pd.Series:

    signals = predictions.copy()

    buy_conflict = (predictions == 1) & (ma_trend == -1)
    sell_conflict = (predictions == -1) & (ma_trend == 1)

    signals[buy_conflict] = 0
    signals[sell_conflict] = 0

    return signals


if __name__ == "__main__":
    predictions = pd.Series(
        model.predict(X),
        index=X.index,
        name="prediction"
    )

    ma_trend = X["ma_trend"]

    final_signals = generate_signals(
        predictions,
        ma_trend
    )
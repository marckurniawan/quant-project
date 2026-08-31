import streamlit as st
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from src.data.data_loader import load_and_validate_data
from src.models.training import prepare_training_data
from src.models.persistence import load_model
from src.backtest.engine import backtest_signals
from src.backtest.metrics import calculate_expectancy, calculate_sharpe_ratio

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def get_model(path: str):
    return load_model(path)

@st.cache_data
def get_data(ticker: str, start: str, end: str):
    return load_and_validate_data(ticker=ticker, start=start, end=end)

@st.cache_data
def get_features(df: pd.DataFrame, threshold: float):
    return prepare_training_data(df, threshold=threshold)

# --- Main app ---

# Load configuration, model and market data
config = load_config()
model = get_model("outputs/model.pkl")
df = get_data(config["ticker"], config["data"]["start_date"], config["data"]["end_date"])

# Prepare features and generate predictions
dataset = get_features(df, config["model"]["threshold"])

X = dataset.drop(columns=["target"])

# Match the index of predictions with X
predictions = pd.Series(model.predict(X), index=X.index, name="prediction")

# Extract signals
buy_prediction = predictions[predictions == 1]
buy_signals = df.loc[buy_prediction.index, ["Close"]]

sell_prediction = predictions[predictions == -1]
sell_signals = df.loc[sell_prediction.index, ["Close"]]

# Dashboard header
st.title("BBCA.JK Trading Signal Dashboard")

st.subheader("Quantitative trading system")
st.markdown("This is a quantitative trading system project designed to generate trading signals based on Machine Learning. The strategy is evaluated using walk-forward validation to preserve the chronological order of the market data.")

# Key metrics
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Rows", value=len(dataset))
col2.metric(label="Ticker", value=config["ticker"])
col3.metric(label="Threshold", value=config["model"]["threshold"])

# Select date range
start_date = st.date_input("Start Date",value=df.index.min().date())

end_date = st.date_input("End Date",value=df.index.max().date())

# Filter price and trading signals according to date range
filtered_df = df[
    (df.index >= pd.Timestamp(start_date)) &
    (df.index <= pd.Timestamp(end_date))]


filtered_buy_signals = buy_signals[
    (buy_signals.index >= pd.Timestamp(start_date)) &
    (buy_signals.index <= pd.Timestamp(end_date))
]

filtered_sell_signals = sell_signals[
    (sell_signals.index >= pd.Timestamp(start_date)) &
    (sell_signals.index <= pd.Timestamp(end_date))
]

# Create price chart with signals
fig, ax = plt.subplots()
ax.plot(filtered_df.index,filtered_df["Close"],label="Close Price")

ax.scatter(filtered_buy_signals.index,filtered_buy_signals["Close"],color="green", marker="^",label="BUY")

ax.scatter(filtered_sell_signals.index,filtered_sell_signals["Close"], color="red", marker="v",label="SELL")

ax.set_title("BBCA.JK Trading Signals")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

# Chart visualization
st.pyplot(fig)

backtest_data = pd.DataFrame({"signal": predictions, "Close": df["Close"]})
backtest_data = backtest_data.dropna()
strategy_returns = backtest_signals(backtest_data, buy_fee=config["backtest"]["buy_fee"], sell_fee=config["backtest"]["sell_fee"])
strategy_returns = pd.Series(strategy_returns)

duration = (df.index.max() - df.index.min()).days / 365.25
total_trades = len(strategy_returns)
trades_per_year = total_trades / duration

strategy_expectancy = calculate_expectancy(strategy_returns)
strategy_sharpe = calculate_sharpe_ratio(strategy_returns, trades_per_year=trades_per_year)

buy_hold_returns = pd.Series(df["Close"]).pct_change().dropna()
buy_hold_sharpe = calculate_sharpe_ratio(buy_hold_returns, trades_per_year=252)

st.subheader("Backtest Performance")
performance_col1, performance_col2, performance_col3 = st.columns(3) 

performance_col1.metric(label="Total Trades", value=total_trades)
performance_col2.metric(label="Expectancy per Trade", value=f"{strategy_expectancy:.2%}")
performance_col3.metric(label="Sharpe Ratio", 
                         value=f"{strategy_sharpe:.2f}", 
                         delta=f"{strategy_sharpe - buy_hold_sharpe:+.2f} vs Buy & Hold")

st.subheader("Feature Importance")

feature_importance = pd.Series(model.feature_importances_,index=X.columns).sort_values()

fig, ax = plt.subplots()

ax.barh(feature_importance.index,feature_importance.values)

ax.set_xlabel("Importance")
ax.set_title("Feature Importance")

st.pyplot(fig)
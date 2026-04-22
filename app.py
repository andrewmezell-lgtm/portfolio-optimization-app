import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Portfolio Optimization App", layout="wide")

st.title("📊 Portfolio Optimization & Risk Analysis")
st.write(
    "This app uses historical stock data and mean-variance optimization to identify an optimal portfolio based on the Sharpe ratio."
)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("User Inputs")

tickers_input = st.sidebar.text_input(
    "Enter stock tickers separated by commas",
    "AAPL,MSFT,NVDA,SPY"
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.04, step=0.005)
num_portfolios = st.sidebar.slider("Simulations", 1000, 5000, 2000, 500)

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)

    if data.empty:
        return pd.DataFrame()

    # Handle both single-ticker and multi-ticker cases
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].copy()
        if len(tickers) == 1:
            prices.columns = [tickers[0]]
        else:
            prices.columns = ["Close"]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices.dropna()

# -----------------------------
# Portfolio math
# -----------------------------
def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = np.sum(mean_returns * weights) * 252
    annual_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return annual_return, annual_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    port_return, port_vol = portfolio_performance(weights, mean_returns, cov_matrix)
    if port_vol == 0:
        return 9999
    return -((port_return - risk_free_rate) / port_vol)

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    initial_guess = num_assets * [1 / num_assets]

    result = minimize(
        negative_sharpe_ratio,
        initial_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    return result

# -----------------------------
# Main app
# -----------------------------
try:
    prices = load_data(tickers, start_date, end_date)

    if prices.empty:
        st.error("No data found. Try different tickers or a different date range.")
    else:
        returns = prices.pct_change().dropna()

        if returns.empty:
            st.error("Returns could not be calculated. Try different inputs.")
        else:
            mean_returns = returns.mean()
            cov_matrix = returns.cov()

            # -----------------------------
            # Raw data
            # -----------------------------
            st.subheader("Price Data")
            st.dataframe(prices.tail())

            st.subheader("Returns")
            st.dataframe(returns.tail())

            st.subheader("Correlation Matrix")
            st.dataframe(returns.corr())

            # -----------------------------
            # Optimization
            # -----------------------------
            result = optimize_portfolio(mean_returns, cov_matrix, risk_free_rate)

            weights = result.x
            ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
            sharpe = (ret - risk_free_rate) / vol if vol != 0 else np.nan

            # -----------------------------
            # Optimal weights table
            # -----------------------------
            st.subheader("Optimal Portfolio Weights")

            weights_df = pd.DataFrame({
                "Asset": prices.columns,
                "Weight": weights
            })

            st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}))

            # -----------------------------
            # Portfolio summary table
            # -----------------------------
            st.subheader("Portfolio Summary")

            summary_df = pd.DataFrame({
                "Metric": ["Expected Return", "Volatility", "Sharpe Ratio"],
                "Value": [ret, vol, sharpe]
            })

            # Format return and volatility as percentages, Sharpe as decimal
            summary_display = summary_df.copy()
            summary_display.loc[summary_display["Metric"] == "Expected Return", "Value"] = f"{ret:.2%}"
            summary_display.loc[summary_display["Metric"] == "Volatility", "Value"] = f"{vol:.2%}"
            summary_display.loc[summary_display["Metric"] == "Sharpe Ratio", "Value"] = f"{sharpe:.2f}"

            st.dataframe(summary_display, use_container_width=True)

            # -----------------------------
            # Efficient frontier simulation
            # -----------------------------
            st.subheader("Efficient Frontier")

            results = np.zeros((3, num_portfolios))

            for i in range(num_portfolios):
                random_weights = np.random.random(len(prices.columns))
                random_weights /= np.sum(random_weights)

                rand_return, rand_vol = portfolio_performance(
                    random_weights, mean_returns, cov_matrix
                )

                results[0, i] = rand_vol
                results[1, i] = rand_return
                results[2, i] = (
                    (rand_return - risk_free_rate) / rand_vol if rand_vol != 0 else np.nan
                )

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                results[0, :],
                results[1, :],
                c=results[2, :],
                alpha=0.6
            )

            ax.scatter(
                vol,
                ret,
                marker="*",
                s=300,
                label="Optimal Portfolio"
            )

            ax.set_title("Efficient Frontier")
            ax.set_xlabel("Annualized Volatility")
            ax.set_ylabel("Annualized Return")
            ax.legend()
            fig.colorbar(scatter, ax=ax, label="Sharpe Ratio")

            st.pyplot(fig)

            # -----------------------------
            # Cumulative returns chart
            # -----------------------------
            st.subheader("Cumulative Returns")

            cumulative_returns = (1 + returns).cumprod()

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            cumulative_returns.plot(ax=ax2)
            ax2.set_title("Growth of $1")
            ax2.set_ylabel("Portfolio Value")
            ax2.set_xlabel("Date")

            st.pyplot(fig2)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)
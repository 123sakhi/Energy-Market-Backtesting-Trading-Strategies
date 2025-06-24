# Energy Market Backtesting & Trading Strategies

This project is an interactive web-based application developed using **Dash** that provides an in-depth analysis of energy market data. The application allows users to forecast energy prices, generate buy/sell signals based on moving averages, and assess key risk metrics like Sharpe Ratio, Value at Risk (VaR), and maximum drawdown. It uses **ARIMA forecasting**, moving averages, and other trading strategies to analyze energy price fluctuations and trading performance.

## Features

- **Energy Price Forecasting** using ARIMA
- **Buy/Sell Signals** based on moving average crossovers
- **Energy Demand vs Available Units** Bar Chart
- **Profit & Loss (P&L)** analysis for trading strategies
- **Hedging Performance** analysis
- **Risk Metrics** such as:
  - Sharpe Ratio
  - Value at Risk (VaR)
  - Max Drawdown
  - Win Rate
- **Market Sentiment** analysis (Bullish, Bearish, Neutral)
- **Interactive Filters & Sliders** for energy units, forecast window, and risk metrics

## Dataset

data/southern_region_energy_market.csv — Dataset used in the analysis. This file contains the energy transaction data for the Southern region market.
## Setup Instructions


1. **Clone the repository:**

   ```bash
   git clone https://github.com/123sakhi/energy-market-backtesting-trading-strategies.git
   cd energy-market-backtesting-trading-strategies

   - `data/southern_region_energy_market.csv` — Dataset used in the analysis. This file contains the energy transaction data for the Southern region market.

2. **Install the required dependencies:**
  ```bash
pip install -r requirements.txt

3. **Run the application:**
  ```bash
python App.py


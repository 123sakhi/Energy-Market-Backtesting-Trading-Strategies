import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import os
import numpy as np
from dash import dash_table

from statsmodels.tsa.arima.model import ARIMA

# Loading and cleaning the dataset
def load_data():
    file_path = r'C:\Users\hp\OneDrive\Desktop\Energy_Dashboard\Assets\energy_transaction_data.csv'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    df = pd.read_csv(file_path)

    # Clean column names and data
    df.columns = df.columns.str.strip().str.lower()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['energy_units_kwh'] = df['energy_units_kwh'].fillna(df['energy_units_kwh'].median())
    df['price_per_kwh'] = df['price_per_kwh'].fillna(df['price_per_kwh'].mean())
    df['available_units'] = df['available_units'].fillna(df['available_units'].median())
    df['demand_units'] = df['demand_units'].fillna(df['demand_units'].median())
    df['energy_demand_kwh'] = df['energy_demand_kwh'].fillna(df['energy_demand_kwh'].median())
    df.drop_duplicates(inplace=True)
    return df

# Example usage:
try:
    df = load_data()
except FileNotFoundError as e:
    print(str(e))

# Group data by hour, converting Period to string
df['hour'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

# ARIMA Model Forecasting
def forecast_arima(df, forecast_steps):
    model = ARIMA(df['price_per_kwh'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_steps)
    forecast_index = pd.date_range(df.index[-1], periods=forecast_steps+1, freq='H')[1:]
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecast'])
    return forecast_df

# Calculate moving averages and signals on hourly data
short_window = 20  # 20-hour moving average
long_window = 50   # 50-hour moving average

df['short_mavg'] = df['price_per_kwh'].rolling(window=short_window, min_periods=1).mean()
df['long_mavg'] = df['price_per_kwh'].rolling(window=long_window, min_periods=1).mean()

# Generate Buy/Sell signals based on moving average crossovers
df['signal'] = 0  # Default no signal
df.loc[short_window:, 'signal'] = np.where(df['short_mavg'][short_window:] > df['long_mavg'][short_window:], 1, 0)  # Buy signal when short MA crosses above long MA
df['positions'] = df['signal'].diff()  # Identify the crossover points (Buy/Sell signals)

# Calculate Hourly Returns for Sharpe Ratio
df['hourly_returns'] = df['price_per_kwh'].pct_change()

# Sharpe Ratio Calculation based on Hourly Returns
sharpe_ratio = df['hourly_returns'].mean() / df['hourly_returns'].std() * np.sqrt(24 * 252)  # Annualized Sharpe Ratio for hourly data (252 trading days)

# Hedging: Calculate the hedging performance
df['hedging_performance'] = df['price_per_kwh'].pct_change() * df['energy_units_kwh']  # Simple hedging performance based on price change and energy units

# Profit and Loss (P&L) metrics: Calculate profit on each trade
df['pnl'] = df['price_per_kwh'].diff() * df['energy_units_kwh']  # Simple P&L calculation assuming units traded

# Total cost calculation assuming a fixed cost per unit (e.g., cost of electricity purchase, handling, etc.)
cost_per_kwh = 0.05  # Example cost per kWh
df['total_cost'] = df['energy_units_kwh'] * cost_per_kwh

# Market Sentiment (Bullish/Bearish)
latest_position = df['positions'].iloc[-1]  # Get the latest position
market_sentiment = "Bullish" if latest_position == 1 else "Bearish" if latest_position == -1 else "Neutral"

# Calculate the Value at Risk (VaR) at a 95% confidence level
VaR = df['pnl'].quantile(0.05) # 5th percentile of PnL as VaR

# Max Drawdown Calculation
df['cumulative_returns'] = (1 + df['hourly_returns']).cumprod()
df['running_max'] = df['cumulative_returns'].cummax()
df['drawdown'] = df['cumulative_returns'] / df['running_max'] - 1
max_drawdown = df['drawdown'].min()  # Max Drawdown

# Win Rate Calculation
win_rate = (df['pnl'] > 0).mean() * 100  # Percentage of profitable trades

# Create the line chart for Price per kWh
fig_price = go.Figure()

fig_price.add_trace(go.Scatter(x=df['hour'], y=df['price_per_kwh'],
                         mode='lines+markers', name='Price per kWh'))

fig_price.update_layout(
    title="Price per kWh with Buy/Sell Signals",
    xaxis_title="Hour",
    yaxis_title="Price per kWh",
    template="plotly_dark"
)

# Create the Short-Term and Long-Term Moving Averages graph with Buy/Sell signals
fig_mavg = go.Figure()

# Plot Short and Long Moving Averages
fig_mavg.add_trace(go.Scatter(x=df['hour'], y=df['short_mavg'], 
                         mode='lines', name='Short-term Moving Average', line=dict(dash='dash')))
fig_mavg.add_trace(go.Scatter(x=df['hour'], y=df['long_mavg'], 
                         mode='lines', name='Long-term Moving Average', line=dict(dash='dash')))

# Plot Buy signals (Green markers when short MA crosses above long MA)
fig_mavg.add_trace(go.Scatter(x=df['hour'][df['positions'] == 1], 
                         y=df['short_mavg'][df['positions'] == 1], 
                         mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'))

# Plot Sell signals (Red markers when short MA crosses below long MA)
fig_mavg.add_trace(go.Scatter(x=df['hour'][df['positions'] == -1], 
                         y=df['short_mavg'][df['positions'] == -1], 
                         mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))

fig_mavg.update_layout(
    datarevision=0,  # Ensure the graph updates when data changes   
    title="Moving Averages with Buy/Sell Signals",
    xaxis_title="Hour",
    yaxis_title="Price per kWh",
    template="plotly_dark")

def update_forecast(forecast_steps):
    # Generate forecast data
    forecast_df = forecast_arima(df, forecast_steps)
    
    # Create the forecast plot
    fig = go.Figure()

    # Plot original price data
    fig.add_trace(go.Scatter(x=df['hour'], y=df['price_per_kwh'], mode='lines', name='Price per kWh'))

    # Plot forecasted values
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Forecast'], mode='lines', name='Forecast', line=dict(color='red')))

    fig.update_layout(
        title=f'Energy Price Forecast for Next {forecast_steps} Hours',
        xaxis_title='Time',
        yaxis_title='Price per kWh',
        template="plotly_dark"
    )

    return fig

# Create the Bar Chart for Energy Demand vs Available Units
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(x=df['hour'], y=df['energy_demand_kwh'], name='Energy Demand (kWh)', marker_color='blue'))
fig_bar.add_trace(go.Bar(x=df['hour'], y=df['available_units'], name='Available Units', marker_color='orange'))

fig_bar.update_layout(
    title="Energy Demand vs Available Units",
    xaxis_title="Hour",
    yaxis_title="Units",
    barmode='group',
    template="plotly_dark"
)

# DataTable for Buy/Sell signals
buy_sell_data = df[df['positions'].notna()][['hour', 'price_per_kwh', 'positions']]
buy_sell_data['Order'] = np.where(buy_sell_data['positions'] == 1, 'Buy', 'Sell')

order_table = dash_table.DataTable(
    id='buy-sell-table',
    columns=[
        {'name': 'Date', 'id': 'hour'},
        {'name': 'Price per kWh', 'id': 'price_per_kwh'},
        {'name': 'Order Type', 'id': 'Order'}
    ],
    data=buy_sell_data.to_dict('records'),
    style_table={'height': '300px', 'overflowY': 'auto'},  # Table with scroll for better usability
    style_cell={'textAlign': 'center', 'padding': '10px'},
    style_header={'backgroundColor': 'rgb(50, 50, 50)', 'fontWeight': 'bold', 'color': 'white'},
    style_data={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'}
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container(
    children=[
        # Header
        dbc.Row([dbc.Col(html.H1("Energy Transaction Dashboard"), className="text-center mb-4", style={'color': 'white'})]),

        # Sharpe Ratio (Performance Metrics)
        dbc.Row([ 
            dbc.Col(html.Div(f"Sharpe Ratio: {sharpe_ratio:.2f}"), width=12, className="text-center mb-4")
        ]),

        # Market Sentiment
        dbc.Row([ 
            dbc.Col(html.Div(f"Market Sentiment: {market_sentiment}"), width=12, className="text-center mb-4")
        ]),

        # Sliders for filtering data (Temperature, Humidity)
        dbc.Row([ 
            dbc.Col(dcc.Slider(id='temperature-slider', min=df['temperature_c'].min(), max=df['temperature_c'].max(), step=0.1, value=df['temperature_c'].min(),
                            marks={i: f"{round(i, 1)}°C" for i in range(int(df['temperature_c'].min()), int(df['temperature_c'].max()) + 1, 5)}, 
                            tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), width=6),
            dbc.Col(dcc.Slider(id='humidity-slider', min=df['humidity_%'].min(), max=df['humidity_%'].max(), step=1, value=df['humidity_%'].min(), 
                            marks={i: f"{i}%" for i in range(int(df['humidity_%'].min()), int(df['humidity_%'].max()) + 1, 5)}, 
                            tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), width=6),
        ], className="mb-4"),

        # Energy Units Slider
        dbc.Row([ 
            dbc.Col(dcc.Slider(id='energy-units-slider', min=df['energy_units_kwh'].min(), max=df['energy_units_kwh'].max(), step=1, value=df['energy_units_kwh'].min(), 
                            marks={i: f"{i} kWh" for i in range(int(df['energy_units_kwh'].min()), int(df['energy_units_kwh'].max()) + 1, 100)}, 
                            tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), width=12)
        ], className="mb-4"),

        # Forecast Slider
        dbc.Row([ 
            dbc.Col(dcc.Slider(id='forecast-slider', min=1, max=72, step=1, value=24, 
                            marks={i: f"{i}h" for i in range(1, 73, 6)}, tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'), width=12)
        ], className="mb-4"),

        # Forecast Graph
        dbc.Row([ 
            dbc.Col(dcc.Graph(id='forecast-graph'), width=12)
        ]),

        # Statistics
        dbc.Row([ 
            dbc.Col(html.Div(f"Average Energy Consumed per Hour: {df['energy_units_kwh'].mean():.2f} kWh"), width=6, className="text-center mb-4"),
            dbc.Col(html.Div(f"Average Price per kWh per Hour: ${df['price_per_kwh'].mean():.2f}"), width=6, className="text-center mb-4")
        ]), 

        # Price & MA Graphs
        dbc.Row([ 
            dbc.Col(dcc.Graph(id='energy-price-graph', figure=fig_price), width=6),
            dbc.Col(dcc.Graph(id='energy-mavg-graph', figure=fig_mavg), width=6),
        ]),

        # Bar Chart
        dbc.Row([ 
            dbc.Col(dcc.Graph(id='energy-bar-chart', figure=fig_bar), width=12),
        ]), 

        # Buy/Sell Order Table
        dbc.Row([ 
            dbc.Col(order_table, width=12),
        ]), 

        # Risk Metrics
        dbc.Row([ 
            dbc.Col(html.Div(f"Value at Risk (VaR): {VaR:.2f}"), width=6, className="text-center mb-4"),
            dbc.Col(html.Div(f"Sharpe Ratio: {sharpe_ratio:.2f}"), width=6, className="text-center mb-4"),
        ]),

        dbc.Row([ 
            dbc.Col(html.Div(f"Max Drawdown: {max_drawdown:.2f}" if not pd.isna(max_drawdown) else "Max Drawdown: N/A"), width=6, className="text-center mb-4"),
            dbc.Col(html.Div(f"Win Rate: {win_rate:.2f}%"), width=6, className="text-center mb-4"),
        ]),
]),
fluid=True  # Optional for full-width responsiveness
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('forecast-slider', 'value')
)
def update_forecast_callback(forecast_steps):
    return update_forecast(forecast_steps)

if __name__ == '__main__':
    app.run(debug=True)
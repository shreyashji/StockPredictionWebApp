# pip install streamlit fbprophet yfinance plotly
import streamlit as st
import numpy as np
import pandas as pds
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2013-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Forecast App')

stocks = ('HDFC.NS', 'TCS.NS', 'INFY.NS', 'ITC.NS','HINDUNILVR.NS','BAJFINANCE.NS','CANBK.NS','COALINDIA.NS','UJJIVANSFB.NS','NMDC.NS','NTPC.NS','POWERGRID.NS','ADANIENT.NS','ADANIGREEN.NS','ADANIPORTS.NS','ADANIPOWER.NS','KOTAKBANK.NS','ICICIBANK.NS','DEEPAKNTR.NS','ASIANPAINT.NS','TATAELXSI.NS')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 8)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

#Candlestick Chart: A candlestick chart provides a visual representation of stock price movements. 
#You can add a candlestick chart to display the high, low, open, and close prices of the stock.
def plot_candlestick_chart():
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.update_layout(title='Candlestick Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_candlestick_chart()
#Volume Chart: Displaying the trading volume of the stock can provide insights into market activity.
# You can add a bar chart to visualize the volume of shares traded.
def plot_volume_chart():
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['Date'], y=data['Volume'], name='Volume'))
    fig.update_layout(title='Volume Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_volume_chart()

#Moving Average Chart: Plotting moving averages can help identify trends and smooth out price fluctuations. 
# You can add a line chart that shows the moving averages for different periods.
def plot_moving_average_chart():
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA_50'], name='MA 50'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA_200'], name='MA 200'))
    fig.update_layout(title='Moving Average Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_moving_average_chart()

# Displaying the predicted closing prices
def plot_predicted_closing_prices():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted'))
    fig.layout.update(title_text='Actual vs. Predicted Closing Prices', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_predicted_closing_prices()

# Displaying the predicted trend and changepoints
def plot_predicted_trend():
    fig = plot_plotly(m, forecast, trend=True)
    fig.layout.update(title_text='Predicted Trend and Changepoints')
    st.plotly_chart(fig)

plot_predicted_trend()

# Displaying the forecast uncertainty
def plot_forecast_uncertainty():
    fig = plot_plotly(m, forecast, uncertainty=True)
    fig.layout.update(title_text='Forecast Uncertainty')
    st.plotly_chart(fig)

plot_forecast_uncertainty()

#In this updated code, the buy threshold is set as 2% below the last available closing price, 
# and the sell threshold is set as 2% above the last available closing price. 
# These thresholds are then plotted as dashed lines on the graph. Green represents the buy threshold, and red represents the sell threshold.
# Add suggestion on when to buy or sell
def add_buy_sell_suggestion():
    last_price = data['Close'].iloc[-1]
    predicted_price = forecast['yhat'].iloc[-1]
    buy_threshold = last_price * 0.98  # 2% below last price
    sell_threshold = last_price * 1.02  # 2% above last price

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted'))

    fig.add_hline(y=buy_threshold, line_dash='dash', line_color='green', name='Buy Threshold')
    fig.add_hline(y=sell_threshold, line_dash='dash', line_color='red', name='Sell Threshold')

    fig.layout.update(title_text='Actual vs. Predicted Closing Prices with Buy/Sell Thresholds',
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

add_buy_sell_suggestion()

#Moving Average Convergence Divergence (MACD) Chart: MACD is a popular technical indicator
#  that helps identify potential buy and sell signals. 
# You can add a chart that displays the MACD line, signal line, and histogram.

# MACD Chart
def plot_macd_chart():
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=macd, name='MACD'))
    fig.add_trace(go.Scatter(x=data['Date'], y=signal, name='Signal'))
    fig.add_trace(go.Bar(x=data['Date'], y=histogram, name='Histogram', marker_color='gray'))
    fig.layout.update(title_text='MACD Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_macd_chart()

#Relative Strength Index (RSI) Chart: RSI is another common technical 
# indicator that measures the momentum of price movements. 
# You can add a chart that shows the RSI values and indicates overbought and oversold levels.
# RSI Chart
def plot_rsi_chart():
    delta = data['Close'].diff()
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = abs(loss.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=rsi, name='RSI'))
    fig.add_hline(y=70, line_dash='dash', line_color='red', name='Overbought')
    fig.add_hline(y=30, line_dash='dash', line_color='green', name='Oversold')
    fig.layout.update(title_text='RSI Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_rsi_chart()

#Bollinger Bands Chart: Bollinger Bands are used to visualize price volatility and potential 
# breakouts. You can add a chart that displays the upper and lower Bollinger Bands along with 
# the stock's price.
# Bollinger Bands Chart
def plot_bollinger_bands_chart():
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=data['Date'], y=rolling_mean, name='Rolling Mean'))
    fig.add_trace(go.Scatter(x=data['Date'], y=upper_band, name='Upper Band', line=dict(color='red', width=1.5)))
    fig.add_trace(go.Scatter(x=data['Date'], y=lower_band, name='Lower Band', line=dict(color='red', width=1.5)))
    fig.layout.update(title_text='Bollinger Bands Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_bollinger_bands_chart()
#Volume Weighted Average Price (VWAP) Chart: VWAP is a trading benchmark that represents the 
# average price at which a stock is traded during the day. You can add a line chart 
# that shows the VWAP values along with the stock's price.
# VWAP Chart
def plot_vwap_chart():
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    cumulative_tp = typical_price.cumsum()
    cumulative_volume = data['Volume'].cumsum()
    vwap = cumulative_tp / cumulative_volume
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Actual'))
    fig.add_trace(go.Scatter(x=data['Date'], y=vwap, name='VWAP'))
    fig.layout.update(title_text='VWAP Chart', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_vwap_chart()

#This code takes the forecasted prices (yhat, yhat_lower, yhat_upper) from the Prophet model
#  and uses them to plot a candlestick chart. Green color is used for increasing prices, 
# and red color is used for decreasing prices. The chart visualizes the predicted opening, 
# high, low, and closing prices.
# Predicted Candlestick Chart
def plot_predicted_candlestick_chart():
    predicted_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(period)
    predicted_data['Date'] = predicted_data['ds']
    predicted_data = predicted_data.set_index('Date')

    fig = go.Figure(data=[go.Candlestick(
        x=predicted_data.index,
        open=predicted_data['yhat_lower'],
        high=predicted_data['yhat_upper'],
        low=predicted_data['yhat_lower'],
        close=predicted_data['yhat'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.layout.update(title_text='Predicted Candlestick Chart')
    st.plotly_chart(fig)

plot_predicted_candlestick_chart()


# Heatmap of Daily Returns
def plot_daily_returns_heatmap():
    daily_returns = data['Close'].pct_change()
    daily_returns = daily_returns[1:].values.reshape(-1, 1)
    sns.heatmap(daily_returns, cmap='RdYlGn', annot=True, fmt=".2%")
    plt.title('Heatmap of Daily Returns')
    plt.xlabel('Days')
    plt.ylabel('Dates')
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)

plot_daily_returns_heatmap()

# Candlestick Chart with Moving Averages and Predictions
def plot_candlestick_with_moving_averages():
    fig = go.Figure(data=[go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])

    fig.add_trace(go.Candlestick(
        x=forecast['ds'],
        open=forecast['yhat_lower'],
        high=forecast['yhat_upper'],
        low=forecast['yhat_lower'],
        close=forecast['yhat'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Predicted'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'].rolling(window=50).mean(),
        mode='lines',
        name='50-day Moving Average'
    ))

    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Close'].rolling(window=200).mean(),
        mode='lines',
        name='200-day Moving Average'
    ))

    fig.layout.update(title_text='Candlestick Chart with Moving Averages and Predictions', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_candlestick_with_moving_averages()

# Volume Chart with Predictions
def plot_volume_chart():
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=data['Date'],
        y=data['Volume'],
        name='Volume',
        marker_color='blue'
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Predicted',
        line=dict(color='red', width=1)
    ))

    fig.layout.update(title_text='Volume Chart with Predictions', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_volume_chart()

# Distribution of Daily Returns with Predictions
def plot_daily_returns_distribution():
    daily_returns = data['Close'].pct_change().dropna()
    sns.histplot(daily_returns, kde=True)

    plt.axvline(x=forecast['yhat'].iloc[-1], color='red', linestyle='--', label='Prediction')
    plt.title('Distribution of Daily Returns with Prediction')
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    fig = plt.gcf()  # Get the current figure
    st.pyplot(fig)

plot_daily_returns_distribution()

# Generate bullet chart
def plot_bullet_chart():
    actual = data['Close'].iloc[-1]
    target = forecast['yhat'].iloc[-1]
    predicted = forecast['yhat'].values[-1]

    fig, ax = plt.subplots(figsize=(10, 6))

    ranges = [target, actual]
    colors = ['lightgray', 'skyblue']
    labels = ['Target', 'Actual']

    ax.barh(y=labels, width=ranges, color=colors)
    ax.axvline(x=predicted, color='red', linestyle='--', linewidth=2)
    ax.set_xlim(0, max(target, actual, predicted) * 1.2)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.2f}'))

    ax.set_xlabel('Value')
    ax.set_ylabel('Category')
    ax.set_title('Bullet Chart')

    st.pyplot(fig)

# Generate bullet chart
plot_bullet_chart()
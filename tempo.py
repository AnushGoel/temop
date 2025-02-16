import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# TensorFlow & Keras for LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
vader_analyzer = SentimentIntensityAnalyzer()

# ----- Helper Functions -----

def get_company_info(ticker):
    """Retrieve company name and description."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName', ticker)
        description = info.get('longBusinessSummary', "No description available.")
        return name, description
    except Exception:
        return ticker, "No description available."

def get_sentiment_from_news(ticker):
    """Calculate an aggregated sentiment score from recent news using VADER."""
    stock = yf.Ticker(ticker)
    news = stock.news
    sentiments = []
    if news and isinstance(news, list):
        for item in news:
            title = item.get('title', '')
            sentiments.append(vader_analyzer.polarity_scores(title)['compound'])
    return float(np.mean(sentiments)) if sentiments else 0.0

def get_historical_data(ticker, start_date, end_date, interval):
    """Retrieve historical data using yfinance."""
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

# ----- Plotting Functions -----

def plot_historical_line(data, ticker):
    """Graph 1: Historical Closing Prices (Line Chart)"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'],
                             mode='lines', name='Close Price'))
    fig.update_layout(title=f"Historical Closing Prices for {ticker}",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

def plot_technical_indicators(data, ticker):
    """Graph 2: Technical Indicators (SMA & EMA) with Closing Prices"""
    data = data.copy()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], mode='lines', name='EMA 20'))
    fig.update_layout(title=f"Technical Indicators for {ticker}",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

def plot_candlestick(data, ticker):
    """Graph 3: Candlestick Chart"""
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'],
                                         name='Candlestick')])
    fig.update_layout(title=f"Candlestick Chart for {ticker}",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

def plot_forecast(data, forecast, ticker, interval):
    """Graph 4: Forecast Overlay (Historical + LSTM Forecast)"""
    # Determine frequency for forecast dates
    if interval == "1d":
        freq = "B"  # Business day frequency for daily data
    else:
        freq = "60min"  # Hourly intervals
    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq=freq)[1:]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast'))
    fig.update_layout(title=f"Forecast for {ticker}",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)
    
    # Display forecast results as a table with dates
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecasted Price": forecast})
    st.write("### Forecast Results with Dates")
    st.dataframe(forecast_df)

# ----- LSTM Model Functions -----

def prepare_data(data, sequence_length):
    """Prepare data for LSTM training."""
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, scaled_data

def build_lstm_model(input_shape):
    """Build and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_lstm(model, data_scaled, scaler, sequence_length, forecast_horizon):
    """Forecast future prices using the trained LSTM model."""
    forecast = []
    last_sequence = data_scaled[-sequence_length:]
    current_sequence = last_sequence.copy()
    for _ in range(forecast_horizon):
        current_sequence_reshaped = np.reshape(current_sequence, (1, sequence_length, 1))
        prediction = model.predict(current_sequence_reshaped)
        forecast.append(prediction[0,0])
        current_sequence = np.append(current_sequence[1:], prediction[0,0])
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
    return forecast.flatten()

# ----- Streamlit App UI -----

st.title("Advanced Interactive Stock Forecasting with Multiple Graphs")
st.write("""
This app displays:
- **Graph 1:** Historical Closing Prices (Line Chart)
- **Graph 2:** Technical Indicators (SMA & EMA)
- **Graph 3:** Candlestick Chart
- **Graph 4:** Forecast Overlay using an LSTM model with forecast results (dates included)
""")

# Sidebar: Input Parameters
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
interval_option = st.sidebar.selectbox("Data Interval", ["1d", "60m"])
forecast_horizon = st.sidebar.number_input("Forecast Horizon (# of intervals)", min_value=1, value=5, step=1)
sequence_length = st.sidebar.number_input("LSTM Sequence Length", min_value=10, value=60, step=1)

if ticker:
    # Display Company Info
    company_name, company_desc = get_company_info(ticker)
    st.subheader(f"Company: {company_name}")
    st.write(company_desc)
    
    if st.button("Refresh Data"):
        st.experimental_rerun()
        
    # Display Enhanced Sentiment Score
    sentiment = get_sentiment_from_news(ticker)
    st.write(f"**Enhanced Sentiment Score from News:** {sentiment:.2f}")
    
    # Fetch historical data
    data = get_historical_data(ticker, start_date.strftime("%Y-%m-%d"),
                               end_date.strftime("%Y-%m-%d"), interval_option)
    
    if data.empty:
        st.error("No historical data found. Adjust your dates or ticker.")
    else:
        # Graph 1: Historical Closing Prices
        st.subheader("Graph 1: Historical Closing Prices")
        plot_historical_line(data, ticker)
        
        # Graph 2: Technical Indicators (SMA 20 & EMA 20)
        st.subheader("Graph 2: Technical Indicators (SMA 20 & EMA 20)")
        plot_technical_indicators(data, ticker)
        
        # Graph 3: Candlestick Chart
        st.subheader("Graph 3: Candlestick Chart")
        plot_candlestick(data, ticker)
        
        # LSTM Model Forecasting and Graph 4: Forecast Overlay
        st.subheader("Graph 4: Forecast Overlay (LSTM Forecast)")
        if len(data) < sequence_length:
            st.error("Not enough data for the specified sequence length.")
        else:
            X, y, scaler, data_scaled = prepare_data(data, sequence_length)
            model = build_lstm_model((X.shape[1], 1))
            with st.spinner("Training LSTM model (this may take a moment)..."):
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            st.success("LSTM model trained!")
            forecast_values = forecast_lstm(model, data_scaled, scaler, sequence_length, int(forecast_horizon))
            st.write(f"Forecast for the next {forecast_horizon} interval(s):")
            st.write(forecast_values)
            plot_forecast(data, forecast_values, ticker, interval_option)

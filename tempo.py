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

# Enhanced Sentiment Analysis: try Hugging Face pipeline; fallback to VADER
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
except Exception:
    sentiment_pipeline = None

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
vader_analyzer = SentimentIntensityAnalyzer()

# ----- Helper Functions -----

def get_company_info(ticker):
    """Return company name and description using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName', ticker)
        description = info.get('longBusinessSummary', "No description available.")
        return name, description
    except Exception:
        return ticker, "No description available."

def get_sentiment_from_news(ticker):
    """
    Compute an aggregated sentiment score from recent news headlines.
    Uses Hugging Face's transformer pipeline if available, else falls back to VADER.
    """
    stock = yf.Ticker(ticker)
    news = stock.news  # List of news items
    sentiments = []
    if news and isinstance(news, list):
        for item in news:
            title = item.get('title', '')
            if sentiment_pipeline:
                try:
                    result = sentiment_pipeline(title)
                    # Map label to a numeric score: POSITIVE => score, NEGATIVE => -score
                    label = result[0]['label'].upper()
                    score = result[0]['score']
                    sentiments.append(score if label == "POSITIVE" else -score)
                except Exception:
                    sentiments.append(vader_analyzer.polarity_scores(title)['compound'])
            else:
                sentiments.append(vader_analyzer.polarity_scores(title)['compound'])
    return float(np.mean(sentiments)) if sentiments else 0.0

def get_historical_data(ticker, start_date, end_date, interval):
    """
    Retrieve historical data using yfinance.
    Interval can be "1d" for daily or "60m" for hourly.
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

def plot_historical_data(data, ticker):
    """Plot interactive historical closing prices using Plotly."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.update_layout(title=f"Historical Closing Prices for {ticker}",
                      xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)

# ----- LSTM Model Functions -----

def prepare_data(data, sequence_length):
    """
    Prepare data for LSTM:
      - Uses the 'Close' column
      - Scales data between 0 and 1
      - Creates sliding window sequences of length `sequence_length`
    """
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    # Reshape for LSTM: (samples, timesteps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm_model(input_shape):
    """Build a simple LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_lstm(model, data_scaled, scaler, sequence_length, forecast_horizon):
    """
    Forecast future prices using the trained LSTM model.
    Iteratively predicts the next value and appends it to the sequence.
    """
    forecast = []
    last_sequence = data_scaled[-sequence_length:]  # Last sequence from training data
    current_sequence = last_sequence.copy()
    
    for _ in range(forecast_horizon):
        # Reshape to (1, sequence_length, 1)
        prediction = model.predict(current_sequence.reshape(1, sequence_length, 1))
        forecast.append(prediction[0, 0])
        # Update the sequence for the next prediction
        current_sequence = np.append(current_sequence[1:], prediction[0, 0])
    # Inverse transform to original scale
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    return forecast.flatten()

# ----- Streamlit App UI -----

st.title("Advanced Interactive Stock Forecasting")

st.write("""
This application provides:
- **Historical graphs** with customizable date ranges.
- **Enhanced sentiment analysis** from recent news.
- **Real-time data refresh**.
- **LSTM model forecasting** (per hour or per day).
- **Company information**.
""")

# Sidebar inputs for customization
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
interval_option = st.sidebar.selectbox("Data Interval", ["1d", "60m"])  # daily or hourly
forecast_horizon = st.sidebar.number_input("Forecast Horizon (# of intervals)", min_value=1, value=5, step=1)
sequence_length = st.sidebar.number_input("LSTM Sequence Length", min_value=10, value=60, step=1)

if ticker:
    # Display company info
    company_name, company_desc = get_company_info(ticker)
    st.subheader(f"Company: {company_name}")
    st.write(company_desc)
    
    # Real-time refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()
    
    # Enhanced sentiment analysis
    sentiment = get_sentiment_from_news(ticker)
    st.write(f"**Enhanced Sentiment Score from News:** {sentiment:.2f}")
    
    # Fetch historical data
    data = get_historical_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), interval_option)
    
    if data.empty:
        st.error("No historical data found. Please adjust your dates or ticker.")
    else:
        st.subheader("Historical Data")
        st.write(data.tail())
        plot_historical_data(data, ticker)
        
        # ----- LSTM Forecasting -----
        st.subheader("LSTM Model Forecasting")
        
        if len(data) < sequence_length:
            st.error("Not enough data for the specified sequence length.")
        else:
            # Prepare data for LSTM
            X, y, scaler = prepare_data(data, sequence_length)
            model = build_lstm_model((X.shape[1], 1))
            
            with st.spinner("Training LSTM model (this may take a moment)..."):
                model.fit(X, y, epochs=10, batch_size=32, verbose=0)
            st.success("LSTM model trained!")
            
            # Forecast future prices
            # We use the scaled 'Close' values for forecasting
            data_scaled = scaler.transform(data['Close'].values.reshape(-1, 1))
            forecast_values = forecast_lstm(model, data_scaled, scaler, sequence_length, int(forecast_horizon))
            st.write(f"Forecast for next {forecast_horizon} interval(s):")
            st.write(forecast_values)
            
            # Plot historical data with forecast
            if interval_option == "1d":
                freq = "B"  # Business day frequency
            else:
                freq = "60min"
            forecast_dates = pd.date_range(start=data.index[-1], periods=forecast_horizon + 1, freq=freq)[1:]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecast'))
            fig.update_layout(title="Historical & Forecasted Prices",
                              xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig)

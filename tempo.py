import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt

# TensorFlow & Keras for LSTM model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon', quiet=True)
vader_analyzer = SentimentIntensityAnalyzer()

# ---------------------------
# Helper Functions
# ---------------------------

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

# ---------------------------
# LSTM Model Functions
# ---------------------------

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
        forecast.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction[0, 0])
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
    return forecast.flatten()

# ---------------------------
# Investment Recommendation
# ---------------------------

def get_investment_recommendation(last_actual, forecast_df, threshold=3):
    """
    Scan the forecast dataframe for the first date on which the forecasted price exceeds
    last_actual by the threshold (percent) and recommend that date.
    """
    recommended_date = None
    for idx, row in forecast_df.iterrows():
        if row['Forecasted Price'] > last_actual * (1 + threshold/100):
            recommended_date = row['Date']
            break
    if recommended_date:
        return f"Based on the forecast, it is recommended to invest on {recommended_date}."
    else:
        return "The forecast does not show a significant upward trend. It might be best to wait."

# ---------------------------
# Altair Chart Functions
# ---------------------------

def chart_historical_line(data, ticker):
    """Graph 1: Historical Closing Prices (Line Chart) using Altair."""
    df = data.reset_index()
    chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title='Close Price')
    ).properties(
        title=f"Historical Closing Prices for {ticker}",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_technical_indicators(data, ticker):
    """Graph 2: Technical Indicators (SMA & EMA) using Altair."""
    df = data.copy().reset_index()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    base = alt.Chart(df).encode(x=alt.X('Date:T', title='Date'))
    line_close = base.mark_line(color='black').encode(y=alt.Y('Close:Q', title='Price'))
    line_sma = base.mark_line(color='blue').encode(y='SMA_20:Q')
    line_ema = base.mark_line(color='red').encode(y='EMA_20:Q')
    chart = (line_close + line_sma + line_ema).properties(
        title=f"Technical Indicators for {ticker}",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_candlestick(data, ticker):
    """Graph 3: Candlestick Chart using Altair."""
    df = data.reset_index()
    base = alt.Chart(df).encode(
        x=alt.X('Date:T', title='Date')
    )
    rule = base.mark_rule().encode(
        y='Low:Q',
        y2='High:Q'
    )
    bar = base.mark_bar().encode(
        y='Open:Q',
        y2='Close:Q',
        color=alt.condition("datum.Open <= datum.Close", alt.value("green"), alt.value("red"))
    )
    chart = (rule + bar).properties(
        title=f"Candlestick Chart for {ticker}",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_forecast_overlay(data, forecast, ticker, interval):
    """
    Graph 4: Forecast Overlay using Altair.
    Overlays historical data with forecasted data.
    """
    # Prepare historical data
    df_hist = data.reset_index()[['Date', 'Close']]
    # Determine frequency and date format based on interval
    if interval == "1d":
        freq = "B"  # Business days
        date_format = "%Y-%m-%d"
    else:
        freq = "60min"
        date_format = "%Y-%m-%d %H:%M"
    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq=freq)[1:]
    forecast_dates = forecast_dates.strftime(date_format)
    df_forecast = pd.DataFrame({
        'Date': pd.to_datetime(forecast_dates),
        'Forecasted Price': forecast
    })
    chart_hist = alt.Chart(df_hist).mark_line(color='black').encode(
        x='Date:T',
        y='Close:Q'
    )
    chart_forecast = alt.Chart(df_forecast).mark_line(color='orange').encode(
        x='Date:T',
        y='Forecasted Price:Q'
    )
    chart = (chart_hist + chart_forecast).properties(
        title=f"Forecast Overlay for {ticker}",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    st.write("### Forecast Results")
    st.dataframe(df_forecast)

# ---------------------------
# Streamlit App UI
# ---------------------------

st.title("Advanced Interactive Stock Forecasting")

st.write("""
This app displays:
- **Graph 1:** Historical Closing Prices (Line Chart)
- **Graph 2:** Technical Indicators (SMA & EMA)
- **Graph 3:** Candlestick Chart
- **Graph 4:** Forecast Overlay (LSTM Forecast)
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
    # Display company info and sentiment
    company_name, company_desc = get_company_info(ticker)
    st.subheader(f"Company: {company_name}")
    st.write(company_desc)
    if st.button("Refresh Data"):
        st.experimental_rerun()
    sentiment = get_sentiment_from_news(ticker)
    st.write(f"**Enhanced Sentiment Score from News:** {sentiment:.2f}")
    
    # Fetch historical data
    data = get_historical_data(
        ticker,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        interval_option
    )
    
    if data.empty:
        st.error("No historical data found. Adjust your dates or ticker.")
    else:
        # Graph 1: Historical Closing Prices
        st.subheader("Graph 1: Historical Closing Prices")
        chart_historical_line(data, ticker)
        
        # Graph 2: Technical Indicators (SMA & EMA)
        st.subheader("Graph 2: Technical Indicators (SMA & EMA)")
        chart_technical_indicators(data, ticker)
        
        # Graph 3: Candlestick Chart
        st.subheader("Graph 3: Candlestick Chart")
        chart_candlestick(data, ticker)
        
        # Graph 4: Forecast Overlay (LSTM Forecast)
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
            chart_forecast_overlay(data, forecast_values, ticker, interval_option)
            
            # Build forecast dataframe for recommendation
            if interval_option == "1d":
                freq = "B"
                date_format = "%Y-%m-%d"
            else:
                freq = "60min"
                date_format = "%Y-%m-%d %H:%M"
            forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast_values)+1, freq=freq)[1:]
            forecast_dates = forecast_dates.strftime(date_format)
            df_forecast = pd.DataFrame({
                'Date': forecast_dates,
                'Forecasted Price': forecast_values
            })
            last_actual = float(data['Close'].iloc[-1])
            recommendation = get_investment_recommendation(last_actual, df_forecast)
            st.write("### Investment Recommendation")
            st.write(recommendation)

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

# For dynamic currency conversion
from forex_python.converter import CurrencyRates

# ---------------------------
# Dynamic Currency Conversion
# ---------------------------
def get_conversion_rate(to_currency):
    """Fetch the latest conversion rate from USD to the target currency."""
    c = CurrencyRates()
    try:
        if to_currency == "USD":
            return 1.0
        else:
            rate = c.get_rate("USD", to_currency)
            return rate
    except Exception:
        # If error, silently default to 1.0 (USD)
        return 1.0

# Currency symbols mapping
currency_symbols = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "INR": "₹"
}

# ---------------------------
# Custom CSS for polished look
# ---------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f5f5f5; }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf, #2e7bcf); color: white; }
    h1, h2, h3, h4 { color: #2e7bcf; }
    </style>
    """,
    unsafe_allow_html=True,
)

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

def get_news_impact(ticker):
    """Return a short message on the likely impact of news on the stock price."""
    sentiment = get_sentiment_from_news(ticker)
    if sentiment > 0.1:
        return f"News appears generally positive (avg sentiment {sentiment:.2f}); this may push prices higher."
    elif sentiment < -0.1:
        return f"News appears generally negative (avg sentiment {sentiment:.2f}); this may put downward pressure on prices."
    else:
        return f"News sentiment is neutral (avg sentiment {sentiment:.2f}); expect little immediate impact."

def get_historical_data(ticker, start_date, end_date, interval="1d"):
    """Retrieve historical data using yfinance (daily data only)."""
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
    return np.round(forecast.flatten(), 2)

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
        return f"Based on the forecast, consider investing on {recommended_date}."
    else:
        return "No significant upward trend detected in the forecast. It might be best to wait."

# ---------------------------
# Altair Chart Functions
# ---------------------------
def chart_historical_line(data, ticker, curr_symbol):
    """Line Chart for Historical Closing Prices."""
    df = data.reset_index()
    chart = alt.Chart(df).mark_line(color="#2e7bcf").encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title=f'Close Price ({curr_symbol})')
    ).properties(
        title=f"{ticker} - Historical Closing Prices",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_technical_indicators(data, ticker, curr_symbol):
    """Line Chart for Technical Indicators (SMA & EMA)."""
    df = data.copy().reset_index()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    base = alt.Chart(df).encode(x=alt.X('Date:T', title='Date'))
    line_close = base.mark_line(color='black').encode(y=alt.Y('Close:Q', title=f'Price ({curr_symbol})'))
    line_sma = base.mark_line(color='blue').encode(y='SMA_20:Q')
    line_ema = base.mark_line(color='red').encode(y='EMA_20:Q')
    chart = (line_close + line_sma + line_ema).properties(
        title=f"{ticker} - Technical Indicators",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_candlestick(data, ticker, curr_symbol):
    """Candlestick Chart."""
    df = data.reset_index()
    base = alt.Chart(df).encode(x=alt.X('Date:T', title='Date'))
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
        title=f"{ticker} - Candlestick Chart ({curr_symbol})",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_forecast_overlay(data, forecast, ticker, interval, curr_symbol):
    """
    Overlay chart of historical data and forecast.
    Also displays a forecast table.
    """
    # For daily data only, use "B" frequency.
    freq = "B"
    date_format = "%Y-%m-%d"
    df_hist = data.reset_index()[['Date', 'Close']]
    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq=freq)[1:]
    forecast_dates = forecast_dates.strftime(date_format)
    df_forecast = pd.DataFrame({
        'Date': pd.to_datetime(forecast_dates),
        'Forecasted Price': forecast
    })
    chart_hist = alt.Chart(df_hist).mark_line(color='black').encode(
        x='Date:T',
        y=alt.Y('Close:Q', title=f'Price ({curr_symbol})')
    )
    chart_forecast = alt.Chart(df_forecast).mark_line(color='orange').encode(
        x='Date:T',
        y=alt.Y('Forecasted Price:Q', title=f'Price ({curr_symbol})')
    )
    chart = (chart_hist + chart_forecast).properties(
        title=f"{ticker} - Forecast Overlay",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    st.write("Forecast Table")
    st.dataframe(df_forecast)

# ---------------------------
# Fundamental Info Display
# ---------------------------
def display_fundamentals(ticker, conv_factor, curr_symbol):
    """Display key stock fundamentals in the chosen currency."""
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        "Current Price": info.get("regularMarketPrice"),
        "Previous Close": info.get("previousClose"),
        "Open": info.get("open"),
        "Day's High": info.get("dayHigh"),
        "Day's Low": info.get("dayLow"),
        "Volume": info.get("volume"),
        "Market Cap": info.get("marketCap")
    }
    for key, value in fundamentals.items():
        if isinstance(value, (int, float)):
            fundamentals[key] = f"{curr_symbol}{round(value * conv_factor, 2)}"
    df_fund = pd.DataFrame(fundamentals, index=[ticker])
    st.dataframe(df_fund)

# ---------------------------
# Forecast Period Mapping
# ---------------------------
forecast_period_mapping = {
    "15 days": 15,
    "1 week": 7,
    "1 month": 22,
    "6 months": 130,
    "1 year": 252
}

# ---------------------------
# Chat Option (Simple)
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def add_chat_message(user, message):
    st.session_state["chat_history"].append((user, message))

def display_chat():
    for user, message in st.session_state["chat_history"]:
        if user == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Advisor:** {message}")

def process_chat(query):
    query = query.lower()
    # A very simple rule-based response.
    if "best stock" in query:
        response = "Based on current trends, AAPL, MSFT, and TSLA are popular choices—but always do your own research!"
    else:
        response = "I'm not sure. Please consult a financial advisor for personalized advice."
    return response

# ---------------------------
# Streamlit Dashboard UI with Tabs
# ---------------------------
st.sidebar.title("Stock Dashboard Settings")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", "AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
# Only daily data is used now.
interval_option = "1d"
forecast_period_option = st.sidebar.selectbox("Forecast Period", list(forecast_period_mapping.keys()))
# Use LSTM sequence length input as before.
sequence_length = st.sidebar.number_input("LSTM Sequence Length", min_value=10, value=60, step=1)
currency = st.sidebar.selectbox("Select Currency", ["USD", "EUR", "GBP", "INR"])
conv_factor = get_conversion_rate(currency)
curr_symbol = currency_symbols.get(currency, "$")

tabs = st.tabs(["Dashboard", "Charts", "Forecast", "Fundamentals", "Chat"])

if ticker:
    # --- Dashboard Tab ---
    with tabs[0]:
        comp_name, comp_desc = get_company_info(ticker)
        st.subheader(comp_name)
        st.write(comp_desc)
        sentiment = get_sentiment_from_news(ticker)
        st.metric(label="News Sentiment Score", value=f"{sentiment:.2f}")
        news_impact = get_news_impact(ticker)
        st.write("News Impact Analysis:")
        st.info(news_impact)
    
    # Fetch historical data and apply currency conversion
    data = get_historical_data(ticker, start_date.strftime("%Y-%m-%d"),
                               end_date.strftime("%Y-%m-%d"), interval_option)
    if data.empty:
        st.error("No historical data found. Please adjust your dates or ticker.")
    else:
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in data.columns:
                data[col] = data[col] * conv_factor

        # --- Charts Tab ---
        with tabs[1]:
            chart_historical_line(data, ticker, curr_symbol)
            chart_technical_indicators(data, ticker, curr_symbol)
            chart_candlestick(data, ticker, curr_symbol)
        
        # --- Forecast Tab ---
        with tabs[2]:
            forecast_horizon = forecast_period_mapping[forecast_period_option]
            if len(data) < sequence_length:
                st.error("Not enough data for the specified sequence length for forecasting.")
            else:
                X, y, scaler, data_scaled = prepare_data(data, sequence_length)
                model = build_lstm_model((X.shape[1], 1))
                with st.spinner("Training LSTM model..."):
                    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                st.success("LSTM model trained!")
                forecast_values = forecast_lstm(model, data_scaled, scaler, sequence_length, forecast_horizon)
                st.write(f"Forecast for the next {forecast_period_option}:")
                st.write(forecast_values)
                chart_forecast_overlay(data, forecast_values, ticker, interval_option, curr_symbol)
                # Build forecast table for recommendation
                freq = "B"
                date_format = "%Y-%m-%d"
                forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast_values)+1, freq=freq)[1:]
                forecast_dates = forecast_dates.strftime(date_format)
                df_forecast = pd.DataFrame({
                    'Date': forecast_dates,
                    'Forecasted Price': forecast_values
                })
                last_actual = float(data['Close'].iloc[-1])
                recommendation = get_investment_recommendation(last_actual, df_forecast)
                st.write("Investment Recommendation:")
                st.info(recommendation)
        
        # --- Fundamentals Tab ---
        with tabs[3]:
            st.write(f"Displaying fundamentals in {currency} ({curr_symbol}):")
            display_fundamentals(ticker, conv_factor, curr_symbol)
    
    # --- Chat Tab ---
    with tabs[4]:
        st.subheader("Ask Your Investment Advisor")
        display_chat()
        user_query = st.text_input("Enter your question:")
        if st.button("Send"):
            if user_query:
                add_chat_message("User", user_query)
                answer = process_chat(user_query)
                add_chat_message("Advisor", answer)
                st.experimental_rerun()

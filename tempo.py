import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import random

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
# Custom CSS for a polished, fun look
# ---------------------------
st.markdown(
    """
    <style>
    .main { background-color: #f0f8ff; }
    .sidebar .sidebar-content { background-image: linear-gradient(#FF5733, #FF8D1A); color: white; }
    h1, h2, h3, h4 { color: #FF5733; }
    .fun-header { font-family: 'Comic Sans MS', cursive, sans-serif; font-size: 2.5em; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper Functions
# ---------------------------
def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName', ticker)
        description = info.get('longBusinessSummary', "No description available.")
        return name, description
    except Exception:
        return ticker, "No description available."

def get_sentiment_from_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news
    sentiments = []
    if news and isinstance(news, list):
        for item in news:
            title = item.get('title', '')
            sentiments.append(vader_analyzer.polarity_scores(title)['compound'])
    return float(np.mean(sentiments)) if sentiments else 0.0

def get_news_impact(ticker):
    sentiment = get_sentiment_from_news(ticker)
    if sentiment > 0.1:
        return f"Good vibes! The news is generally positive (avg sentiment {sentiment:.2f})."
    elif sentiment < -0.1:
        return f"Uh-oh! The news is leaning negative (avg sentiment {sentiment:.2f})."
    else:
        return f"The news feels pretty neutral (avg sentiment {sentiment:.2f})."

def get_historical_data(ticker, start_date, end_date, interval="1d"):
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

# ---------------------------
# LSTM Model Functions
# ---------------------------
def prepare_data(data, sequence_length):
    dataset = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, data

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_lstm(model, data_scaled, scaler, sequence_length, forecast_horizon):
    forecast = []
    last_sequence = data_scaled[-sequence_length:]
    current_sequence = last_sequence.copy()
    for _ in range(forecast_horizon):
        current_sequence_reshaped = np.reshape(current_sequence, (1, sequence_length, 1))
        prediction = model.predict(current_sequence_reshaped, verbose=0)
        forecast.append(prediction[0, 0])
        current_sequence = np.append(current_sequence[1:], prediction[0, 0])
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1,1))
    return np.round(forecast.flatten(), 2)

# ---------------------------
# Investment Recommendation
# ---------------------------
def get_investment_recommendation(last_actual, forecast_df, threshold=3):
    recommended_date = None
    for idx, row in forecast_df.iterrows():
        if row['Forecasted Price'] > last_actual * (1 + threshold/100):
            recommended_date = row['Date'].strftime("%Y-%m-%d")
            break
    if recommended_date:
        return f"Great news! Consider investing on {recommended_date}."
    else:
        return "No significant upward trend detected. Maybe hold off for now."

# ---------------------------
# Altair Chart Functions
# ---------------------------
def chart_historical_line(data, ticker):
    df = data.reset_index()
    chart = alt.Chart(df).mark_line(color="#FF5733").encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title='Close Price (USD)')
    ).properties(
        title=f"{ticker} - Historical Closing Prices",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_technical_indicators(data, ticker):
    df = data.copy().reset_index()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    base = alt.Chart(df).encode(x=alt.X('Date:T', title='Date'))
    line_close = base.mark_line(color='black').encode(y=alt.Y('Close:Q', title='Price (USD)'))
    line_sma = base.mark_line(color='blue').encode(y='SMA_20:Q')
    line_ema = base.mark_line(color='red').encode(y='EMA_20:Q')
    chart = (line_close + line_sma + line_ema).properties(
        title=f"{ticker} - Technical Indicators",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_candlestick(data, ticker):
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
        title=f"{ticker} - Candlestick Chart (USD)",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_volume_bar(data, ticker):
    df = data.reset_index()
    chart = alt.Chart(df).mark_bar(color="#FF8D1A").encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Volume:Q', title='Volume')
    ).properties(
        title=f"{ticker} - Trading Volume",
        width=700,
        height=300
    )
    st.altair_chart(chart, use_container_width=True)

def chart_forecast_overlay(data, forecast, ticker):
    freq = "B"  # Business days
    date_format = "%Y-%m-%d"
    df_hist = data.reset_index()[['Date', 'Close']]
    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq=freq)[1:]
    df_forecast = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Price': forecast
    })
    chart_hist = alt.Chart(df_hist).mark_line(color='black').encode(
        x='Date:T',
        y=alt.Y('Close:Q', title='Price (USD)')
    )
    chart_forecast = alt.Chart(df_forecast).mark_line(color='orange').encode(
        x='Date:T',
        y=alt.Y('Forecasted Price:Q', title='Price (USD)')
    )
    chart = (chart_hist + chart_forecast).properties(
        title=f"{ticker} - Forecast Overlay",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)
    st.write("Forecast Table")
    st.dataframe(df_forecast)
    return df_forecast

# ---------------------------
# Fundamental Info Display
# ---------------------------
def display_fundamentals(ticker):
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
    df_fund = pd.DataFrame(fundamentals, index=[ticker])
    st.dataframe(df_fund)

# ---------------------------
# Chat Option and Processing
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
    if "clear chat" in query:
        st.session_state["chat_history"] = []
        return "Chat cleared!"
    elif "closing price" in query:
        if "apple" in query or "aapl" in query:
            ticker = "AAPL"
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get("regularMarketPrice", None)
            if price is not None:
                return f"Apple's closing price today is ${round(price, 2)}."
            else:
                return "I couldn't fetch the closing price at this time."
        else:
            return "I'm not sure about that. Please consult a financial advisor for personalized advice."
    elif "best stock" in query:
        return "Based on current trends, AAPL, MSFT, and TSLA are popular choices—but always do your own research!"
    else:
        return "I'm not sure. Please consult a financial advisor for personalized advice."

# ---------------------------
# Sidebar Inputs and Tabs Setup
# ---------------------------
st.sidebar.title("Stock Dashboard Settings")

# Use session_state to store the ticker so that Surprise Me works properly
if "ticker_input" not in st.session_state:
    st.session_state["ticker_input"] = "AAPL"

ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", st.session_state["ticker_input"], key="ticker_input").upper().strip()
if st.sidebar.button("Surprise Me"):
    surprise_ticker = random.choice(["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "NFLX", "NVDA"])
    st.session_state["ticker_input"] = surprise_ticker
    ticker = surprise_ticker

start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
interval_option = "1d"
forecast_period_type = st.sidebar.selectbox("Forecast Period Type", 
                                              ["Days", "Weeks", "15 Days", "Months", "6 Months", "Year"])
number_of_periods = st.sidebar.number_input("Number of Periods", min_value=1, value=1, step=1)
total_forecast_days = number_of_periods * {
    "Days": 1,
    "Weeks": 7,
    "15 Days": 15,
    "Months": 22,
    "6 Months": 130,
    "Year": 252
}[forecast_period_type]
sequence_length = st.sidebar.number_input("LSTM Sequence Length", min_value=10, value=60, step=1)

tabs = st.tabs(["Dashboard", "Charts", "Forecast", "Fundamentals", "Chat"])

# ---------------------------
# Dashboard Tab
# ---------------------------
if ticker:
    with tabs[0]:
        st.markdown("<h1 class='fun-header'>Welcome to the Stock Fun House!</h1>", unsafe_allow_html=True)
        comp_name, comp_desc = get_company_info(ticker)
        st.subheader(comp_name)
        st.write(comp_desc)
        with st.expander("More Company Info"):
            try:
                stock_info = yf.Ticker(ticker).info
                st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                st.write(f"**Country:** {stock_info.get('country', 'N/A')}")
                st.write(f"**CEO:** {stock_info.get('ceo', 'N/A')}")
                st.write(f"**Website:** {stock_info.get('website', 'N/A')}")
            except Exception:
                st.write("Additional company information not available.")
                
        sentiment = get_sentiment_from_news(ticker)
        st.metric(label="News Sentiment Score", value=f"{sentiment:.2f}")
        news_impact = get_news_impact(ticker)
        st.write("News Impact Analysis:")
        st.info(news_impact)
    
    # ---------------------------
    # Fetch Historical Data (USD only)
    # ---------------------------
    data = get_historical_data(ticker, start_date.strftime("%Y-%m-%d"),
                               end_date.strftime("%Y-%m-%d"), interval_option)
    if data.empty:
        st.error("No historical data found. Please adjust your dates or ticker.")
    else:
        # ---------------------------
        # Charts Tab
        # ---------------------------
        with tabs[1]:
            st.subheader("Let's Visualize Some Data!")
            chart_historical_line(data, ticker)
            chart_technical_indicators(data, ticker)
            chart_candlestick(data, ticker)
            chart_volume_bar(data, ticker)
        
        # ---------------------------
        # Forecast Tab
        # ---------------------------
        with tabs[2]:
            st.subheader("Price Forecast Fun!")
            if len(data) < sequence_length:
                st.error("Not enough data for the specified sequence length for forecasting.")
            else:
                X, y, scaler, data_used = prepare_data(data, sequence_length)
                model = build_lstm_model((X.shape[1], 1))
                with st.spinner("Training LSTM model... (this might take a moment!)"):
                    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                st.success("Model trained! Here comes the forecast...")
                forecast_values = forecast_lstm(model, scaler.transform(data['Close'].values.reshape(-1,1)), scaler, sequence_length, total_forecast_days)
                st.write(f"Forecast for the next {total_forecast_days} day(s):")
                st.write(forecast_values)
                df_forecast = chart_forecast_overlay(data, forecast_values, ticker)
                
                last_actual = float(data['Close'].iloc[-1])
                recommendation = get_investment_recommendation(last_actual, df_forecast)
                st.write("Investment Recommendation:")
                st.info(recommendation)
                
                # Prediction Summary
                predicted_growth = ((forecast_values[-1] - last_actual) / last_actual) * 100
                st.metric(label="Predicted Growth (%)", value=f"{predicted_growth:.2f}%")
                with st.expander("Prediction Summary"):
                    if predicted_growth > 0:
                        st.success("The future looks bright! It might be a good time to invest.")
                    else:
                        st.error("The forecast suggests caution. It may be wise to wait.")
                    st.write("Remember: This forecast is for fun and based on historical data. Always do your own research!")
        
        # ---------------------------
        # Fundamentals Tab
        # ---------------------------
        with tabs[3]:
            st.subheader("Key Fundamentals (USD)")
            display_fundamentals(ticker)
    
    # ---------------------------
    # Chat Tab
    # ---------------------------
    with tabs[4]:
        st.subheader("Chat with Your Investment Advisor")
        if st.button("Clear Chat"):
            st.session_state["chat_history"] = []
        display_chat()
        user_query = st.text_input("Enter your question here:")
        if st.button("Send"):
            if user_query:
                add_chat_message("User", user_query)
                answer = process_chat(user_query)
                add_chat_message("Advisor", answer)
        display_chat()

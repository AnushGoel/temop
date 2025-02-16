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

# For GPT Chat integration
import openai

# ---------------------------
# Set currency to USD (fixed)
# ---------------------------
currency = "USD"
conv_factor = 1.0
curr_symbol = "$"

# ---------------------------
# Custom CSS for a polished look
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

def forecast_lstm(model, data_scaled, scaler, sequence_length, forecast_days):
    """Forecast future prices using the trained LSTM model."""
    forecast = []
    last_sequence = data_scaled[-sequence_length:]
    current_sequence = last_sequence.copy()
    for _ in range(forecast_days):
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
def chart_historical_line(data, ticker):
    """Line Chart for Historical Closing Prices."""
    df = data.reset_index()
    chart = alt.Chart(df).mark_line(color="#2e7bcf").encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Close:Q', title='Close Price ($)')
    ).properties(
        title=f"{ticker} - Historical Closing Prices",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_technical_indicators(data, ticker):
    """Line Chart for Technical Indicators (SMA & EMA)."""
    df = data.copy().reset_index()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    base = alt.Chart(df).encode(x=alt.X('Date:T', title='Date'))
    line_close = base.mark_line(color='black').encode(y=alt.Y('Close:Q', title='Price ($)'))
    line_sma = base.mark_line(color='blue').encode(y='SMA_20:Q')
    line_ema = base.mark_line(color='red').encode(y='EMA_20:Q')
    chart = (line_close + line_sma + line_ema).properties(
        title=f"{ticker} - Technical Indicators",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

def chart_candlestick(data, ticker):
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
        title=f"{ticker} - Candlestick Chart ($)",
        width=700,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

# ---------------------------
# Forecast Table Function
# ---------------------------
def display_forecast_table(data, forecast, forecast_days):
    """
    Display a forecast table showing the forecast for the next 'forecast_days' days.
    """
    freq = "B"  # Business days
    date_format = "%Y-%m-%d"
    forecast_dates = pd.date_range(start=data.index[-1], periods=len(forecast)+1, freq=freq)[1:]
    forecast_dates = forecast_dates.strftime(date_format)
    df_forecast = pd.DataFrame({
        'Date': pd.to_datetime(forecast_dates),
        'Forecasted Price': forecast
    })
    st.write("### Forecast Table")
    st.dataframe(df_forecast)
    return df_forecast

# ---------------------------
# Fundamental Info Display
# ---------------------------
def display_fundamentals(ticker):
    """Display key stock fundamentals in USD."""
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
            fundamentals[key] = f"${round(value, 2)}"
    df_fund = pd.DataFrame(fundamentals, index=[ticker])
    st.dataframe(df_fund)

# ---------------------------
# Chat Option and Processing using OpenAI GPT
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
    # Use OpenAI's GPT model if API key is available
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        openai.api_key = api_key
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": query}],
                temperature=0.7
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            return "Sorry, I'm having trouble processing that query right now."
    else:
        # Fallback answer if no API key is provided
        return "OpenAI API key is not set. Please set it in the Streamlit secrets for GPT functionality."

# ---------------------------
# Sidebar Inputs and Tabs Setup
# ---------------------------
st.sidebar.title("Stock Dashboard Settings")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", "AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date.today())
# Daily data only
interval_option = "1d"
forecast_days = st.sidebar.number_input("Number of Forecast Days", min_value=1, value=5, step=1)
sequence_length = st.sidebar.number_input("LSTM Sequence Length", min_value=10, value=60, step=1)

tabs = st.tabs(["Dashboard", "Charts", "Forecast", "Fundamentals", "Chat"])

# ---------------------------
# Dashboard Tab
# ---------------------------
if ticker:
    with tabs[0]:
        comp_name, comp_desc = get_company_info(ticker)
        st.subheader(comp_name)
        st.write(comp_desc)
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
            chart_historical_line(data, ticker)
            chart_technical_indicators(data, ticker)
            chart_candlestick(data, ticker)
        
        # ---------------------------
        # Forecast Tab
        # ---------------------------
        with tabs[2]:
            if len(data) < sequence_length:
                st.error("Not enough data for the specified sequence length for forecasting.")
            else:
                X, y, scaler, data_scaled = prepare_data(data, sequence_length)
                model = build_lstm_model((X.shape[1], 1))
                with st.spinner("Training LSTM model..."):
                    model.fit(X, y, epochs=10, batch_size=32, verbose=0)
                st.success("LSTM model trained!")
                forecast_values = forecast_lstm(model, data_scaled, scaler, sequence_length, forecast_days)
                st.write(f"Forecast for the next {forecast_days} day(s):")
                st.write(forecast_values)
                df_forecast = display_forecast_table(data, forecast_values, forecast_days)
                last_actual = float(data['Close'].iloc[-1])
                recommendation = get_investment_recommendation(last_actual, df_forecast)
                st.write("Investment Recommendation:")
                st.info(recommendation)
        
        # ---------------------------
        # Fundamentals Tab
        # ---------------------------
        with tabs[3]:
            st.write("Displaying fundamentals in USD ($):")
            display_fundamentals(ticker)
    
    # ---------------------------
    # Chat Tab
    # ---------------------------
    with tabs[4]:
        st.subheader("Ask Your Investment Advisor")
        if st.button("Clear Chat"):
            st.session_state["chat_history"] = []
        display_chat()
        user_query = st.text_input("Enter your question:")
        if st.button("Send"):
            if user_query:
                add_chat_message("User", user_query)
                answer = process_chat(user_query)
                add_chat_message("Advisor", answer)
        display_chat()

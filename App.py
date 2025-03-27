import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
import ta  # Technical indicators library
import os

# Define model path
MODEL_PATH = "stock_prediction_model.pkl"

# Check if model file exists before loading
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("⚠️ Model file not found! Please upload 'stock_prediction_model.pkl' before running the app.")
    model = None  # Prevent errors if model is missing

# Function to fetch real-time stock data
def fetch_real_time_stock_data(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="60d", interval="1d")  # Last 60 days of data

    if history.empty:
        return None

    # Calculate indicators
    history['SMA_10'] = ta.trend.sma_indicator(history['Close'], window=10)
    history['SMA_50'] = ta.trend.sma_indicator(history['Close'], window=50)
    history['EMA_50'] = ta.trend.ema_indicator(history['Close'], window=50)
    history['RSI'] = ta.momentum.rsi(history['Close'], window=14)

    macd = ta.trend.MACD(history['Close'])
    history['MACD'] = macd.macd()
    history['MACD_signal'] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(history['Close'], window=20)
    history['Upper_Band'] = bb.bollinger_hband()
    history['Middle_Band'] = bb.bollinger_mavg()
    history['Lower_Band'] = bb.bollinger_lband()

    latest_data = history.iloc[-1]  # Get last row

    # Select required features
    feature_columns = ['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']
    feature_data = latest_data[feature_columns]

    return np.array(feature_data).reshape(1, -1)

# Streamlit UI
st.title("📈 Real-Time Stock Prediction")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., TATAPOWER.NS):", "TATAPOWER.NS")

if st.button("Predict"):
    if model is None:
        st.error("⚠️ Model not loaded. Upload 'stock_prediction_model.pkl'.")
    else:
        latest_data = fetch_real_time_stock_data(ticker)

        if latest_data is None:
            st.warning("⚠️ Error fetching stock data. Try another ticker.")
        else:
            try:
                prediction = model.predict(latest_data)[0]
                st.success(f"**Recommendation:** {'✅ Buy' if prediction == 1 else '❌ Sell'}")

                # Display fetched technical indicators
                df = pd.DataFrame(latest_data, columns=['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band'])
                st.write("### Technical Indicators")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

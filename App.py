import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
with open("stock_model.pkl", "rb") as f:
    model = pickle.load(f)

# Function to fetch real-time stock data
def fetch_real_time_stock_data(ticker):
    stock = yf.Ticker(ticker)
    history = stock.history(period="60d", interval="1d")  # Last 60 days for indicators

    if history.empty:
        return None

    # Calculate indicators
    history['SMA_10'] = ta.sma(history['Close'], length=10)
    history['SMA_50'] = ta.sma(history['Close'], length=50)
    history['EMA_50'] = ta.ema(history['Close'], length=50)
    history['RSI'] = ta.rsi(history['Close'], length=14)
    
    macd = ta.macd(history['Close'])
    history['MACD'] = macd.iloc[:, 0]  # MACD Line
    history['MACD_signal'] = macd.iloc[:, 1]  # Signal Line
    
    bb = ta.bbands(history['Close'], length=20)
    history['Upper_Band'] = bb.iloc[:, 0]
    history['Middle_Band'] = bb.iloc[:, 1]
    history['Lower_Band'] = bb.iloc[:, 2]

    latest_data = history.iloc[-1]  # Get last row

    # Select required features
    feature_data = latest_data[['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']]
    
    return np.array(feature_data).reshape(1, -1)

# Streamlit UI
st.title("📈 Real-Time Stock Prediction")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., TATAPOWER.NS):", "TATAPOWER.NS")

if st.button("Predict"):
    latest_data = fetch_real_time_stock_data(ticker)
    
    if latest_data is None:
        st.error("Error fetching stock data. Try another ticker.")
    else:
        prediction = model.predict(latest_data)[0]
        st.write(f"**Recommendation:** {'✅ Buy' if prediction == 1 else '❌ Sell'}")

        # Display fetched technical indicators
        st.write("### Technical Indicators")
        st.write(pd.DataFrame(latest_data, columns=['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']))

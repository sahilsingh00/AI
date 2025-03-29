import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib

# Load trained model
model = joblib.load("model1.pkl")

# Define technical indicators calculation
def compute_indicators(history):
    history['SMA_10'] = ta.sma(history['Close'], length=10)
    history['SMA_50'] = ta.sma(history['Close'], length=50)
    history['EMA_50'] = ta.ema(history['Close'], length=50)
    history['RSI'] = ta.rsi(history['Close'], length=14)
    
    macd = ta.macd(history['Close'])
    history['MACD'] = macd.iloc[:, 0]
    history['MACD_signal'] = macd.iloc[:, 1]
    
    bb = ta.bbands(history['Close'], length=20)
    history['Upper_Band'] = bb.iloc[:, 0]
    history['Middle_Band'] = bb.iloc[:, 1]
    history['Lower_Band'] = bb.iloc[:, 2]
    
    return history.dropna()

# Function to get stock prediction
def get_recommendation(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    history = stock.history(period="6mo", interval="1d").iloc[-1:]
    
    if history.empty:
        return "No data available for this stock."
    
    history = compute_indicators(history)
    if history.empty:
        return "Not enough data for prediction."
    
    features = ['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']
    input_features = history[features].iloc[-1:].values
    prediction = model.predict(input_features)[0]
    recommendation = "Buy" if prediction == 1 else "Sell"
    
    info = stock.info
    current_price = info.get('currentPrice', 'N/A')
    pe_ratio = info.get('trailingPE', 'N/A')
    market_cap = info.get('marketCap', 'N/A')
    revenue_growth = info.get('revenueGrowth', 'N/A')
    
    return {
        "Recommendation": recommendation,
        "Current Price": f"₹{current_price}",
        "P/E Ratio": pe_ratio,
        "Market Capitalization": market_cap,
        "Revenue Growth": revenue_growth
    }

# Streamlit UI
st.title("Stock Prediction App")
st.write("Enter a stock ticker to get a Buy/Sell recommendation based on technical and fundamental analysis.")

stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TATAPOWER.NS)", "TATAPOWER.NS")

if st.button("Predict"):
    result = get_recommendation(stock_ticker)
    if isinstance(result, str):
        st.warning(result)
    else:
        st.subheader(f"Recommendation: **{result['Recommendation']}**")
        st.write(f"**Current Price:** {result['Current Price']}")
        st.write(f"**P/E Ratio:** {result['P/E Ratio']}")
        st.write(f"**Market Capitalization:** {result['Market Capitalization']}")
        st.write(f"**Revenue Growth:** {result['Revenue Growth']}")

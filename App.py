import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib

# Load stock list CSV
@st.cache_data
def load_stock_list():
    return pd.read_csv("stocklist.csv")

stocks_df = load_stock_list()

# Load trained model
try:
    model = joblib.load("stock_prediction_model.pkl")
except FileNotFoundError:
    st.error("Error: Model file not found. Please upload the trained model.")
    st.stop()

# Function to compute technical indicators
def compute_indicators(history):
    if history.empty or 'Close' not in history.columns:
        return pd.DataFrame()

    history['SMA_10'] = ta.sma(history['Close'], length=10)
    history['SMA_50'] = ta.sma(history['Close'], length=50)
    history['EMA_50'] = ta.ema(history['Close'], length=50)
    history['RSI'] = ta.rsi(history['Close'], length=14)

    macd = ta.macd(history['Close'])
    if macd is not None:
        history['MACD'] = macd.iloc[:, 0]
        history['MACD_signal'] = macd.iloc[:, 1]

    bb = ta.bbands(history['Close'], length=20)
    if bb is not None:
        history['Upper_Band'] = bb.iloc[:, 0]
        history['Middle_Band'] = bb.iloc[:, 1]
        history['Lower_Band'] = bb.iloc[:, 2]

    return history.dropna()

# Function to get stock prediction
def get_recommendation(stock_ticker):
    try:
        stock = yf.Ticker(stock_ticker)
        history = stock.history(period="6mo", interval="1d")

        if history.empty:
            return "No data available for this stock."

        # history = compute_indicators(history.iloc[-1:])
        history = compute_indicators(history)
        features = ['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']
        
        if history.empty or not all(f in history.columns for f in features):
            return "Not enough data for prediction."

        # Select features for model
        # features = ['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']
        
        input_features = history[features].iloc[-1:].values
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        recommendation = "Buy" if prediction == 1 else "Sell"

        # Get fundamental data
        info = stock.info or {}
        current_price = info.get('currentPrice', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        # market_cap = info.get('marketCap', 'N/A')
        revenue_growth = info.get('revenueGrowth', 'N/A')

        def format_market_cap(value):
            if value == 'N/A' or value is None:
                return 'N/A'
            if value >= 100_00_00_000:  # 100 Crore (1 Cr = 10,000,000)
                return f"{value / 100_00_00_000:.2f} Cr"
            else:
                return f"{value / 10_00_000:.2f} Lakh"

        market_cap = format_market_cap(info.get('marketCap', 'N/A'))

        return {
            "Recommendation": recommendation,
            "Current Price": f"₹{current_price}" if current_price != 'N/A' else 'N/A',
            "P/E Ratio": pe_ratio,
            "Market Capitalization": market_cap,
            "Revenue Growth": revenue_growth
        }
    
    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Streamlit UI
st.title("📈 Stock Prediction App")
# st.write("Enter a stock ticker to get a **Buy/Sell recommendation** based on technical and fundamental analysis.")


# Step 1: Select stock from CSV
stock_name = st.selectbox("Select a Stock:", stocks_df["NAME OF COMPANY"].unique())

# Get corresponding symbol
stock_symbol = stocks_df[stocks_df["NAME OF COMPANY"] == stock_name]["SYMBOL"].values[0]

# Step 2: Choose NSE or BSE
market_choice = st.radio("Select Market:", ["NSE", "BSE"])

# Convert to correct ticker format
if market_choice == "NSE":
    stock_ticker = stock_symbol + ".NS"
elif market_choice == "BSE":
    stock_ticker = stock_symbol + ".BO"

# st.write(f"Selected Ticker: `{stock_ticker}`")


# # Input for stock ticker
# stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TATAPOWER.NS)", "TATAPOWER.NS")

if st.button("🔍 Predict"):
    result = get_recommendation(stock_ticker)
    
    if isinstance(result, str):
        st.warning(result)
    else:
        st.subheader(f"📊 Recommendation: **{result['Recommendation']}**")
        st.markdown("---")
        st.table(pd.DataFrame(result.items(), columns=["Metric", "Value"]))

st.markdown("---")
st.markdown(
    "**⚠️ Disclaimer:** This tool provides predictions based on historical data and technical indicators. "
    "It is NOT financial advice. Please conduct your own market research before making any investment decisions. 📉"
)

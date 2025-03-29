# import streamlit as st
# import yfinance as yf
# import pandas as pd
# import pandas_ta as pta
# import numpy as np
# import joblib
# import ta
# # Load trained model
# try:
#     model = joblib.load("stock_prediction_model (1).pkl")
# except FileNotFoundError:
#     st.error("Error: Model file 'model1.pkl' not found. Please upload the trained model.")
#     st.stop()

# # Function to compute technical indicators
# def compute_indicators(history):
#     if history.empty or 'Close' not in history.columns:
#         return pd.DataFrame()  # Return empty DataFrame if data is missing

#     history['SMA_10'] = pta.sma(history['Close'], length=10)
#     history['SMA_50'] = pta.sma(history['Close'], length=50)
#     history['EMA_50'] = pta.ema(history['Close'], length=50)
#     history['RSI'] = pta.rsi(history['Close'], length=14)

#     macd = pta.macd(history['Close'])
#     if macd is not None:
#         history['MACD'] = macd.iloc[:, 0]
#         history['MACD_signal'] = macd.iloc[:, 1]

#     bb = pta.bbands(history['Close'], length=20)
#     if bb is not None:
#         history['Upper_Band'] = bb.iloc[:, 0]
#         history['Middle_Band'] = bb.iloc[:, 1]
#         history['Lower_Band'] = bb.iloc[:, 2]

#     return history.dropna()

# # Function to get stock prediction
# def get_recommendation(stock_ticker):
#     try:
#         stock = yf.Ticker(stock_ticker)
#         history = stock.history(period="6mo", interval="1d")

#         if history.empty:
#             return "No data available for this stock."

#         history = compute_indicators(history.iloc[-1:])
#         if history.empty:
#             return "Not enough data for prediction."

#         # Select features for model
#         features = ['Close', 'SMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_signal', 'Upper_Band', 'Middle_Band', 'Lower_Band']
#         input_features = history[features].iloc[-1:].values
        
#         # Make prediction
#         prediction = model.predict(input_features)[0]
#         recommendation = "Buy" if prediction == 1 else "Sell"

#         # Get fundamental data
#         info = stock.info or {}  # Ensure info is a dictionary
#         current_price = info.get('currentPrice', 'N/A')
#         pe_ratio = info.get('trailingPE', 'N/A')
#         market_cap = info.get('marketCap', 'N/A')
#         revenue_growth = info.get('revenueGrowth', 'N/A')

#         return {
#             "Recommendation": recommendation,
#             "Current Price": f"₹{current_price}" if current_price != 'N/A' else 'N/A',
#             "P/E Ratio": pe_ratio,
#             "Market Capitalization": market_cap,
#             "Revenue Growth": revenue_growth
#         }
    
#     except Exception as e:
#         return f"Error fetching data: {str(e)}"

# # Streamlit UI
# st.title("📈 Stock Prediction App")
# st.write("Enter a stock ticker to get a **Buy/Sell recommendation** based on technical and fundamental analysis.")

# # Input for stock ticker
# stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TATAPOWER.NS)", "TATAPOWER.NS")

# if st.button("🔍 Predict"):
#     result = get_recommendation(stock_ticker)
    
#     if isinstance(result, str):
#         st.warning(result)
#     else:
#         st.subheader(f"📊 Recommendation: **{result['Recommendation']}**")
#         st.markdown("---")
#         st.table(pd.DataFrame(result.items(), columns=["Metric", "Value"]))







import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as pta
import numpy as np
import joblib
import ta
# Load trained model
try:
    model = joblib.load("stock_prediction_model (1).pkl")
except FileNotFoundError:
    st.error("Error: Model file 'model1.pkl' not found. Please upload the trained model.")
    st.stop()

# Function to compute technical indicators
def compute_indicators(history):
    if history.empty or 'Close' not in history.columns:
        return pd.DataFrame()  # Return empty DataFrame if data is missing

    history['SMA_10'] = pta.sma(history['Close'], length=10)
    history['SMA_50'] = pta.sma(history['Close'], length=50)
    history['EMA_50'] = pta.ema(history['Close'], length=50)
    history['RSI'] = pta.rsi(history['Close'], length=14)

    macd = pta.macd(history['Close'])
    if macd is not None:
        history['MACD'] = macd.iloc[:, 0]
        history['MACD_signal'] = macd.iloc[:, 1]

    bb = pta.bbands(history['Close'], length=20)
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
        info = stock.info or {}  # Ensure info is a dictionary
        current_price = info.get('currentPrice', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        revenue_growth = info.get('revenueGrowth', 'N/A')

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
st.write("Enter a stock ticker to get a **Buy/Sell recommendation** based on technical and fundamental analysis.")

# Input for stock ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TATAPOWER.NS)", "TATAPOWER.NS")

if st.button("🔍 Predict"):
    result = get_recommendation(stock_ticker)
    
    if isinstance(result, str):
        st.warning(result)
    else:
        st.subheader(f"📊 Recommendation: **{result['Recommendation']}**")
        st.markdown("---")
        st.table(pd.DataFrame(result.items(), columns=["Metric", "Value"]))

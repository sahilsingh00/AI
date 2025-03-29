
import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Stock Prediction App", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 18px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .stRadio [role=radiogroup] {
            display: flex;
            gap: 20px;
        }
        .stTable {
            text-align: left;
        }
        .highlight {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .warning {
            font-size: 18px;
            font-weight: bold;
            color: #FF5733;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    if history.empty or "Close" not in history.columns:
        return pd.DataFrame()

    history["SMA_10"] = ta.sma(history["Close"], length=10)
    history["SMA_50"] = ta.sma(history["Close"], length=50)
    history["EMA_50"] = ta.ema(history["Close"], length=50)
    history["RSI"] = ta.rsi(history["Close"], length=14)

    macd = ta.macd(history["Close"])
    if macd is not None:
        history["MACD"] = macd.iloc[:, 0]
        history["MACD_signal"] = macd.iloc[:, 1]

    bb = ta.bbands(history["Close"], length=20)
    if bb is not None:
        history["Upper_Band"] = bb.iloc[:, 0]
        history["Middle_Band"] = bb.iloc[:, 1]
        history["Lower_Band"] = bb.iloc[:, 2]

    return history.dropna()

# Function to get stock prediction
def get_recommendation(stock_ticker):
    try:
        stock = yf.Ticker(stock_ticker)
        history = stock.history(period="6mo", interval="1d")

        if history.empty:
            return "No data available for this stock."

        history = compute_indicators(history)
        features = ["Close", "SMA_10", "SMA_50", "EMA_50", "RSI", "MACD", "MACD_signal", "Upper_Band", "Middle_Band", "Lower_Band"]

        if history.empty or not all(f in history.columns for f in features):
            return "Not enough data for prediction."

        input_features = history[features].iloc[-1:].values
        prediction = model.predict(input_features)[0]
        recommendation = "Buy" if prediction == 1 else "Sell"

        # Get fundamental data
        info = stock.info or {}
        current_price = info.get("currentPrice", "N/A")
        pe_ratio = info.get("trailingPE", "N/A")
        revenue_growth = info.get("revenueGrowth", "N/A")

        # Format revenue growth as percentage
        if revenue_growth != "N/A" and isinstance(revenue_growth, (int, float)):
            revenue_growth = f"{revenue_growth * 100:.2f}%"

        # Format market capitalization
        def format_market_cap(value):
            if value == "N/A" or value is None:
                return "N/A"
            if value >= 100_00_00_000:  # 100 Crore (1 Cr = 10,000,000)
                return f"{value / 100_00_00_000:.2f} Cr"
            else:
                return f"{value / 10_00_000:.2f} Lakh"

        market_cap = format_market_cap(info.get("marketCap", "N/A"))

        # Define conclusion
        conclusion = (
            "📈 **The stock is showing strong momentum**, with positive technical signals and favorable fundamentals."
            if recommendation == "Buy"
            else "⚠️ **The stock appears weak**, with potential downside risks. It may not be the right time to invest."
        )

        return {
            "Recommendation": recommendation,
            "Current Price": f"₹{current_price}" if current_price != "N/A" else "N/A",
            "P/E Ratio": pe_ratio,
            "Market Capitalization": market_cap,
            "Revenue Growth": revenue_growth,
            "Conclusion": conclusion,
        }

    except Exception as e:
        return f"Error fetching data: {str(e)}"

# Streamlit UI
st.title("📊 Stock Prediction App")
st.write("🔍 **Select a stock and market to get a Buy/Sell prediction**")

# Step 1: Select stock from CSV
stock_name = st.selectbox("📌 Select a Stock:", stocks_df["NAME OF COMPANY"].unique())

# Get corresponding symbol
stock_symbol = stocks_df[stocks_df["NAME OF COMPANY"] == stock_name]["SYMBOL"].values[0]

# Step 2: Choose NSE or BSE
market_choice = st.radio("🏦 Select Market:", ["NSE", "BSE"], horizontal=True)

# Convert to correct ticker format
stock_ticker = f"{stock_symbol}.NS" if market_choice == "NSE" else f"{stock_symbol}.BO"

# Prediction Button
if st.button("📈 Predict Now"):
    result = get_recommendation(stock_ticker)

    if isinstance(result, str):
        st.warning(result)
    else:
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 **Recommendation: {result['Recommendation']}**")
            if result["Recommendation"] == "Buy":
                st.markdown('<p class="highlight">📈 Strong Momentum!</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="warning">⚠️ Potential Downside Risk</p>', unsafe_allow_html=True)

        with col2:
            st.subheader("💰 **Stock Overview**")
            st.write(f"🔹 **Current Price:** {result['Current Price']}")
            st.write(f"🔹 **P/E Ratio:** {result['P/E Ratio']}")
            st.write(f"🔹 **Market Cap:** {result['Market Capitalization']}")
            st.write(f"🔹 **Revenue Growth:** {result['Revenue Growth']}")

        st.markdown("---")
        st.markdown(f"📝 **Conclusion:** {result['Conclusion']}")

# Disclaimer
st.markdown("---")
st.markdown(
    "**⚠️ Disclaimer:** This tool provides predictions based on historical data and technical indicators. "
    "It is **NOT financial advice**. Please conduct your own market research before making any investment decisions. 📉"
)

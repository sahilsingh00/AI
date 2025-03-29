import streamlit as st

# Example function to generate output
def get_stock_recommendation(ticker):
    # Dummy data (replace with your model's output)
    recommendation = "Buy"
    stock_data = {
        "Current Price": "₹375.40",
        "52-Week High": "₹410.50",
        "52-Week Low": "₹280.30",
        "52-Week Change": "35.0%",
        "Market Capitalization": "₹1.2T",
        "Trailing P/E Ratio": "25.4",
        "Profit Margin": "8.5%",
        "Quarterly Revenue Growth": "12.3%",
        "Recent Developments": "The company is expanding into renewable energy, increasing growth potential."
    }

    return recommendation, stock_data

# Streamlit UI
st.title("Real-Time Stock Prediction")

ticker = st.text_input("Enter Stock Ticker (e.g., TATAPOWER.NS):")
if st.button("Predict"):
    recommendation, stock_data = get_stock_recommendation(ticker)

    # Display Recommendation
    st.markdown(f"### **{recommendation}** {ticker} at this time.")

    # Display Stock Data
    for key, value in stock_data.items():
        st.markdown(f"**{key}:** {value}")

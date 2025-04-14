from typing import ClassVar
from langchain.tools import BaseTool

class StockDataTool(BaseTool):
    name: ClassVar[str] = "Stock Tool"
    description: ClassVar[str] = "Fetches stock data using yfinance and technical indicators."

    def _run(self, query: str) -> str:
        # You can integrate real yfinance code here
        return f"Fetched stock data for: {query}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")
from typing import ClassVar
from langchain.tools import BaseTool
import joblib
import numpy as np
import yfinance as yf
import pandas_ta as ta

class StockPredictionTool(BaseTool):
    name: ClassVar[str] = "Stock Prediction Tool"
    description: ClassVar[str] = "Uses ML model to predict Buy/Hold/Sell for Indian stocks."

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        self.model = joblib.load(model_path)

    def _run(self, query: str) -> str:
        try:
            stock = yf.Ticker(query)
            hist = stock.history(period="6mo", interval="1d")

            hist["EMA_50"] = ta.ema(hist["Close"], length=50)
            hist["SMA_50"] = ta.sma(hist["Close"], length=50)
            hist["RSI"] = ta.rsi(hist["Close"], length=14)
            macd = ta.macd(hist["Close"])
            if macd is not None:
                hist["MACD"] = macd["MACD_12_26_9"]
                hist["MACD_signal"] = macd["MACDs_12_26_9"]
                hist["MACD_hist"] = macd["MACDh_12_26_9"]
            bb = ta.bbands(hist["Close"], length=20)
            if bb is not None:
                hist["Upper_Band"] = bb["BBU_20_2.0"]
                hist["Middle_Band"] = bb["BBM_20_2.0"]
                hist["Lower_Band"] = bb["BBL_20_2.0"]

            hist = hist.dropna()
            features = ['Close', 'EMA_50', 'High', 'Low', 'Lower_Band', 'MACD', 'MACD_hist',
                        'MACD_signal', 'Middle_Band', 'Open', 'RSI', 'SMA_50', 'Upper_Band', 'Volume']
            
            input_features = hist[features].iloc[-1:].values
            prediction = self.model.predict(input_features)[0]
            return ["Sell", "Hold", "Buy"][prediction]
        except Exception as e:
            return f"Prediction Error: {str(e)}"

    def _arun(self, query: str):
        raise NotImplementedError("Async not supported.")

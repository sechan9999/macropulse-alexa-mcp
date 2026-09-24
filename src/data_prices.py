import pandas as pd
import numpy as np
import yfinance as yf

def load_prices(tickers, start="2006-01-01"):
    px = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    px = px.resample("ME").last().dropna()
    return px

def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna()

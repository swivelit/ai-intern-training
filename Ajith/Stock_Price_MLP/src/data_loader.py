import yfinance as yf
import os
import pandas as pd

def download_stock(symbol="AAPL", start="2018-01-01", end="2024-01-01"):
    os.makedirs("data", exist_ok=True)

    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False
    )

    # 🚨 HARD CHECK
    if df is None or df.empty:
        raise RuntimeError(
            f"❌ Failed to download data for {symbol}. "
            "Check internet or try another symbol."
        )

    df.reset_index(inplace=True)
    df.to_csv("data/stock.csv", index=False)

    print(f"✅ Downloaded {symbol} data: {len(df)} rows")
    return df

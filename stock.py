import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
import json

def main():
    stocks = ["TMGH.CA", "COMI.CA", "EFIH.CA"]
    results = []
    
    for ticker in stocks:
        data = yf.download(ticker, period="1mo", interval="1d")
        if data.empty: continue
        
        last_price = float(data['Close'].iloc[-1])
        rsi = float(RSIIndicator(data['Close']).rsi().iloc[-1])
        sma_20 = float(data['Close'].rolling(window=20).mean().iloc[-1])
        
        strategy = "Observation"
        if rsi < 35: strategy = "Buy Zone"
        elif rsi > 65: strategy = "Profit Zone"
        
        results.append({
            "Ticker": ticker,
            "Price": round(last_price, 2),
            "RSI": round(rsi, 2),
            "SMA": round(sma_20, 2),
            "Strategy": strategy
        })
    
    # حفظ النتيجة في ملف JSON
    with open('daily_report.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Report Generated Successfully!")

if __name__ == "__main__":
    main()

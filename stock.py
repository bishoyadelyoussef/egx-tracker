import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
import json

def main():
    # الأكواد الرسمية للأسهم
    stocks = ["TMGH.CA", "COMI.CA", "EFIH.CA"]
    results = []
    
    print("--- Starting EGX Data Fetch ---")
    
    for ticker in stocks:
        try:
            # بننزل بيانات 3 شهور عشان نضمن حساب المتوسطات صح
            data = yf.download(ticker, period="3mo", interval="1d", progress=False)
            
            if data.empty or len(data) < 20:
                print(f"Skipping {ticker}: Not enough data.")
                continue
            
            # معالجة بيانات السعر (بما يتناسب مع سيرفرات لينكس)
            close_prices = data['Close']
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]

            last_price = float(close_prices.iloc[-1])
            rsi = float(RSIIndicator(close_prices).rsi().iloc[-1])
            sma_20 = float(SMAIndicator(close_prices, window=20).sma_indicator().iloc[-1])
            
            # تحديد التوصية
            strategy = "Wait (Neutral)"
            if rsi < 30: strategy = "Buy Zone"
            elif rsi > 70: strategy = "Sell Zone"
            
            results.append({
                "Ticker": ticker,
                "Price": round(last_price, 2),
                "RSI": round(rsi, 2),
                "SMA_20": round(sma_20, 2),
                "Strategy": strategy
            })
            print(f"Successfully processed {ticker}")
        except Exception as e:
            print(f"Error on {ticker}: {e}")

    # حفظ ملف الـ JSON النهائي لـ Flowise
    with open('daily_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"DONE: Report saved with {len(results)} stocks.")

if __name__ == "__main__":
    main()

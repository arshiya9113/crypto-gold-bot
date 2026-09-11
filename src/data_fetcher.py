"""
دریافت داده‌های قیمتی:
- کریپتو: از صرافی بایننس با کتابخانه ccxt (بدون نیاز به API Key برای داده OHLCV)
- طلا: از یاهو فایننس با کتابخانه yfinance (نماد پیش‌فرض GC=F یعنی فیوچرز طلای کامکس)
"""
import time

import ccxt
import pandas as pd
import yfinance as yf

EXCLUDED_PREFIXES = ("UP/", "DOWN/", "BULL/", "BEAR/")


def get_exchange():
    """صرافی بایننس را برای دریافت داده عمومی (بدون نیاز به کلید) آماده می‌کند."""
    return ccxt.binance({"enableRateLimit": True})


def get_top_crypto_symbols(exchange, max_symbols=40, min_quote_volume=5_000_000, quote="USDT"):
    """
    فهرست نمادهای اسپات جفت‌شده با USDT را بر اساس حجم معاملات ۲۴ ساعته مرتب کرده
    و N نماد برتر (پرمعامله‌ترین) را برمی‌گرداند.
    """
    markets = exchange.load_markets()
    tickers = exchange.fetch_tickers()

    candidates = []
    for symbol, market in markets.items():
        if not market.get("spot", True):
            continue
        if not symbol.endswith("/" + quote):
            continue
        if symbol.startswith(EXCLUDED_PREFIXES):
            continue
        ticker = tickers.get(symbol)
        if not ticker:
            continue
        quote_volume = ticker.get("quoteVolume") or 0
        if quote_volume < min_quote_volume:
            continue
        candidates.append((symbol, quote_volume))

    candidates.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in candidates[:max_symbols]]


def fetch_crypto_ohlcv(exchange, symbol, timeframe="1h", limit=300, retries=2):
    """کندل‌های OHLCV یک نماد کریپتو را برمی‌گرداند."""
    for attempt in range(retries + 1):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                print(f"[WARN] دریافت داده {symbol} ناموفق بود: {exc}")
                return None
            time.sleep(1.5)
    return None


def _flatten_yf_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    return data


def fetch_gold_ohlcv(ticker="GC=F", interval="1h", period="60d"):
    """کندل‌های OHLCV طلا را از یاهو فایننس برمی‌گرداند."""
    try:
        data = yf.download(
            ticker, interval=interval, period=period, progress=False, auto_adjust=False
        )
        if data is None or data.empty:
            return None
        data = _flatten_yf_columns(data)
        data = data.reset_index()
        data.columns = [str(c).lower() for c in data.columns]
        date_col = "datetime" if "datetime" in data.columns else "date"
        df = data.rename(columns={date_col: "timestamp"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
        return df.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] دریافت داده طلا ({ticker}) ناموفق بود: {exc}")
        return None

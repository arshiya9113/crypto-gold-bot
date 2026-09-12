"""
دریافت داده‌های قیمتی:
- کریپتو: با کتابخانه ccxt، از یک لیست صرافی امتحان می‌شود تا اولین صرافی که از
  محل اجرای سرور (مثلا سرورهای گیت‌هاب اکشن در آمریکا) در دسترس باشد پیدا شود.
  علت این تصمیم: بایننس جهانی (binance.com) دسترسی از IP آمریکا را طبق قوانین
  خودش مسدود می‌کند (خطای HTTP 451) و چون گیت‌هاب اکشن روی سرورهای آمریکایی
  اجرا می‌شود، صرافی‌های جایگزین/سازگار با آمریکا (مثل Binance.US و Kraken) هم
  در لیست قرار گرفته‌اند تا ربات همیشه بتواند داده بگیرد.
- طلا: از یاهو فایننس با کتابخانه yfinance (نماد پیش‌فرض GC=F یعنی فیوچرز طلای کامکس)
"""
import time

import ccxt
import pandas as pd
import yfinance as yf

LEVERAGED_TOKEN_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


def _is_leveraged_token(symbol):
    """نمادهای اهرمی مثل BTCUP/USDT یا ETHBEAR/USDT را تشخیص می‌دهد."""
    base = symbol.split("/")[0]
    return base.endswith(LEVERAGED_TOKEN_SUFFIXES)


def get_exchange(exchange_id):
    """یک نمونه صرافی ccxt بر اساس شناسه (مثلا 'kraken') می‌سازد."""
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def get_top_crypto_symbols_multi(exchange_ids, max_symbols=40, min_quote_volume=1_000_000,
                                  quotes=("USDT", "USD")):
    """
    صرافی‌های داده‌شده را به ترتیب امتحان می‌کند تا یکی از IP فعلی در دسترس باشد،
    سپس پرمعامله‌ترین نمادهای آن (بر اساس حجم معاملات ۲۴ ساعته) را برمی‌گرداند.
    خروجی: (exchange, symbols) یا (None, []) اگر هیچ صرافی در دسترس نبود.
    """
    for exchange_id in exchange_ids:
        try:
            exchange = get_exchange(exchange_id)
            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] صرافی {exchange_id} در دسترس نیست یا خطا داد: {exc}")
            continue

        candidates = []
        for symbol, market in markets.items():
            if not market.get("spot", True):
                continue
            if market.get("quote") not in quotes:
                continue
            if _is_leveraged_token(symbol):
                continue
            ticker = tickers.get(symbol)
            if not ticker:
                continue
            quote_volume = ticker.get("quoteVolume") or 0
            if quote_volume < min_quote_volume:
                continue
            candidates.append((symbol, quote_volume))

        if not candidates:
            print(f"[WARN] صرافی {exchange_id} متصل شد ولی هیچ نماد واجد شرایطی یافت نشد")
            continue

        candidates.sort(key=lambda item: item[1], reverse=True)
        symbols = [symbol for symbol, _ in candidates[:max_symbols]]
        print(f"[INFO] اتصال موفق به صرافی {exchange_id} ({len(symbols)} نماد یافت شد)")
        return exchange, symbols

    return None, []


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
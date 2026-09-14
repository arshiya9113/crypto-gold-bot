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
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def _base_currency(symbol):
    return symbol.split("/")[0]


def get_multi_exchange_symbols(exchange_ids, max_total_symbols=80, min_quote_volume=1_000_000,
                                quotes=("USDT", "USD"), max_exchanges=3):
    """
    برخلاف حالت قبلی (فقط اولین صرافی در دسترس)، اینجا تا max_exchanges صرافی
    موفق را با هم ترکیب می‌کند تا استخر نمادهای قابل بررسی واقعا بزرگ‌تر شود.
    اگر یک ارز (مثلا BTC) در چند صرافی هم‌زمان واجد شرایط باشد، فقط پرحجم‌ترین
    جفت آن نگه داشته می‌شود (برای جلوگیری از تحلیل تکراری یک دارایی).

    خروجی: لیستی از (symbol, exchange_instance) که هرکدام باید بعدا برای همان
    نماد از همان صرافی داده OHLCV بگیرند، مرتب‌شده بر اساس حجم معاملات نزولی.
    """
    collected = []  # (symbol, quote_volume, exchange_instance)
    successful_exchanges = 0

    for exchange_id in exchange_ids:
        if successful_exchanges >= max_exchanges:
            break
        try:
            exchange = get_exchange(exchange_id)
            markets = exchange.load_markets()
            tickers = exchange.fetch_tickers()
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] صرافی {exchange_id} در دسترس نیست یا خطا داد: {exc}")
            continue

        found_here = 0
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
            collected.append((symbol, quote_volume, exchange))
            found_here += 1

        if found_here > 0:
            successful_exchanges += 1
            print(f"[INFO] اتصال موفق به صرافی {exchange_id} ({found_here} نماد واجد شرایط یافت شد)")
        else:
            print(f"[WARN] صرافی {exchange_id} متصل شد ولی هیچ نماد واجد شرایطی نداشت")

    if not collected:
        return []

    # دیدوپ بر اساس ارز پایه: برای هر ارز (مثلا BTC)، پرحجم‌ترین جفت را نگه دار
    best_by_base = {}
    for symbol, volume, exchange in collected:
        base = _base_currency(symbol)
        if base not in best_by_base or volume > best_by_base[base][1]:
            best_by_base[base] = (symbol, volume, exchange)

    ranked = sorted(best_by_base.values(), key=lambda item: item[1], reverse=True)
    top = ranked[:max_total_symbols]
    return [(symbol, exchange) for symbol, _, exchange in top]


def fetch_historical_ohlcv(exchange, symbol, timeframe="1h", days=120):
    """
    برخلاف fetch_crypto_ohlcv (که فقط آخرین N کندل را می‌گیرد)، این تابع با
    صفحه‌بندی (pagination) از طریق پارامتر since، چند ماه کندل تاریخی پشت‌سرهم
    می‌گیرد - مخصوص بک‌تست، نه اجرای زنده هر ساعت.
    """
    try:
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    except Exception:  # noqa: BLE001
        timeframe_ms = 60 * 60 * 1000  # پیش‌فرض ۱ ساعت اگر صرافی پشتیبانی نکرد

    since = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    all_candles = []
    max_iterations = 200  # سقف ایمنی در برابر حلقه بی‌پایان

    for _ in range(max_iterations):
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] دریافت تاریخچه {symbol} ناموفق بود: {exc}")
            break

        if not batch:
            break

        all_candles.extend(batch)
        last_ts = batch[-1][0]
        next_since = last_ts + timeframe_ms

        if next_since <= since or len(batch) < 2:
            break
        since = next_since

        if since >= exchange.milliseconds():
            break
        time.sleep(max(exchange.rateLimit / 1000, 0.2))

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)  # حذف کندل ناقص/درحال‌شکل‌گیری، مثل حالت زنده

    return df


def fetch_many_crypto_ohlcv(symbol_exchange_pairs, timeframe="1h", limit=300, max_workers=8):
    """
    کندل‌های OHLCV چند نماد را به‌صورت هم‌زمان (با چند Thread) می‌گیرد تا زمان کل
    اجرا به‌جای رشد خطی با تعداد نمادها، به‌شدت کوتاه‌تر شود. چون هر نماد از
    صرافی مخصوص به خودش گرفته می‌شود (خروجی get_multi_exchange_symbols)، فشار
    درخواست بین چند صرافی مختلف هم پخش می‌شود.
    """
    results = {}

    def _worker(item):
        symbol, exchange = item
        return symbol, fetch_crypto_ohlcv(exchange, symbol, timeframe=timeframe, limit=limit)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_worker, item) for item in symbol_exchange_pairs]
        for future in as_completed(futures):
            symbol, df = future.result()
            results[symbol] = df

    return results


def fetch_crypto_ohlcv(exchange, symbol, timeframe="1h", limit=300, retries=2):
    """
    کندل‌های OHLCV یک نماد کریپتو را برمی‌گرداند.
    نکته مهم: آخرین کندلی که صرافی برمی‌گرداند معمولا هنوز کامل نشده (در حال
    شکل‌گیری است) و مقادیرش هر لحظه تغییر می‌کند. برای جلوگیری از سیگنال‌های
    ناپایدار/متناقض و ATR مصنوعا کوچک، این کندل ناقص همیشه از خروجی حذف می‌شود
    و فقط کندل‌های کاملا بسته‌شده برای تحلیل باقی می‌مانند.
    """
    for attempt in range(retries + 1):
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not raw or len(raw) < 2:
                return None
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.iloc[:-1].reset_index(drop=True)  # حذف کندل ناقص/درحال‌شکل‌گیری
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
    """
    کندل‌های OHLCV طلا/نقره را از یاهو فایننس برمی‌گرداند. مثل کریپتو، آخرین
    کندل (ساعت در حال جریان) هنوز کامل نیست و حذف می‌شود تا تحلیل روی داده
    پایدار و کاملا بسته‌شده انجام شود.
    """
    try:
        data = yf.download(
            ticker, interval=interval, period=period, progress=False, auto_adjust=False
        )
        if data is None or data.empty or len(data) < 2:
            return None
        data = _flatten_yf_columns(data)
        data = data.reset_index()
        data.columns = [str(c).lower() for c in data.columns]
        date_col = "datetime" if "datetime" in data.columns else "date"
        df = data.rename(columns={date_col: "timestamp"})
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()
        df = df.iloc[:-1].reset_index(drop=True)  # حذف کندل ناقص/درحال‌شکل‌گیری
        return df
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] دریافت داده طلا ({ticker}) ناموفق بود: {exc}")
        return None

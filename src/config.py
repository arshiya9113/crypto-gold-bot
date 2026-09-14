"""
تمام تنظیمات ربات از طریق متغیرهای محیطی (Environment Variables) خوانده می‌شود
تا بدون تغییر کد، فقط با تنظیم Secrets در گیت‌هاب بشود رفتار ربات را عوض کرد.
"""
import os


def _get_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _get_str(name, default):
    """
    مثل os.environ.get عمل می‌کند با این تفاوت که اگر متغیر محیطی وجود داشته باشد
    ولی مقدارش رشته خالی '' باشد (مثلا یک GitHub Actions Variable که تنظیم نشده)،
    باز هم مقدار پیش‌فرض را برمی‌گرداند.
    """
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


# --- تنظیمات ایمیل ---
EMAIL_SMTP_SERVER = _get_str("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = _get_int("EMAIL_SMTP_PORT", 587)
EMAIL_FROM = _get_str("EMAIL_FROM", "")
EMAIL_PASSWORD = _get_str("EMAIL_PASSWORD", "")  # برای جیمیل: App Password
EMAIL_TO = [e.strip() for e in _get_str("EMAIL_TO", "").split(",") if e.strip()]

# --- تنظیمات بازار کریپتو ---
# لیست صرافی‌ها به ترتیب اولویت امتحان می‌شوند تا اولین صرافی در دسترس (از نظر
# محدودیت جغرافیایی سرور اجراکننده) پیدا شود. شناسه‌ها باید نام صرافی در ccxt باشند.
CRYPTO_EXCHANGES = [
    e.strip() for e in _get_str(
        "CRYPTO_EXCHANGES", "binanceus,kraken,coinbase,kucoin,okx,bybit,gateio,mexc"
    ).split(",") if e.strip()
]
CRYPTO_QUOTE_CURRENCIES = [
    q.strip() for q in _get_str("CRYPTO_QUOTE_CURRENCIES", "USDT,USD").split(",") if q.strip()
]
MAX_CRYPTO_SYMBOLS = _get_int("MAX_CRYPTO_SYMBOLS", 80)          # چند رمزارز برتر (یکتا) بررسی شود
MIN_QUOTE_VOLUME_USDT = _get_float("MIN_QUOTE_VOLUME_USDT", 1_000_000)  # حداقل حجم معاملات ۲۴ ساعته
MAX_EXCHANGES_TO_MERGE = _get_int("MAX_EXCHANGES_TO_MERGE", 3)   # چند صرافی موفق با هم ترکیب شوند
MAX_FETCH_WORKERS = _get_int("MAX_FETCH_WORKERS", 8)             # تعداد Threadهای هم‌زمان برای دریافت کندل
CRYPTO_TIMEFRAME = _get_str("CRYPTO_TIMEFRAME", "1h")            # تایم‌فریم کندل‌ها
CRYPTO_CANDLE_LIMIT = _get_int("CRYPTO_CANDLE_LIMIT", 300)       # تعداد کندل برای تحلیل
TOP_SIGNALS_COUNT = _get_int("TOP_SIGNALS_COUNT", 15)            # چند سیگنال برتر در ایمیل نمایش داده شود

# --- تنظیمات فلزات گران‌بها ---
# GC=F: فیوچرز طلای کامکس | SI=F: فیوچرز نقره کامکس
METAL_TICKERS = [t.strip() for t in _get_str("METAL_TICKERS", "GC=F,SI=F").split(",") if t.strip()]
METAL_LABELS = {"GC=F": "GOLD", "SI=F": "SILVER"}
METAL_INTERVAL = _get_str("METAL_INTERVAL", "1h")
METAL_PERIOD = _get_str("METAL_PERIOD", "60d")

# --- عمومی ---
TIMEZONE = _get_str("TIMEZONE", "Asia/Tehran")
SIGNAL_HISTORY_FILE = _get_str("SIGNAL_HISTORY_FILE", "signal_history.csv")

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
MAX_CRYPTO_SYMBOLS = _get_int("MAX_CRYPTO_SYMBOLS", 60)          # چند رمزارز برتر بررسی شود
MIN_QUOTE_VOLUME_USDT = _get_float("MIN_QUOTE_VOLUME_USDT", 5_000_000)  # حداقل حجم معاملات ۲۴ ساعته
CRYPTO_TIMEFRAME = _get_str("CRYPTO_TIMEFRAME", "1h")            # تایم‌فریم کندل‌ها
CRYPTO_CANDLE_LIMIT = _get_int("CRYPTO_CANDLE_LIMIT", 300)       # تعداد کندل برای تحلیل
TOP_SIGNALS_COUNT = _get_int("TOP_SIGNALS_COUNT", 15)            # چند سیگنال برتر در ایمیل نمایش داده شود

# --- تنظیمات طلا ---
# GC=F: فیوچرز طلای کامکس | می‌توان چند نماد را با کاما جدا کرد، مثلا "GC=F,XAUUSD=X"
GOLD_TICKERS = [t.strip() for t in _get_str("GOLD_TICKERS", "GC=F").split(",") if t.strip()]
GOLD_INTERVAL = _get_str("GOLD_INTERVAL", "1h")
GOLD_PERIOD = _get_str("GOLD_PERIOD", "60d")

# --- عمومی ---
TIMEZONE = _get_str("TIMEZONE", "Asia/Tehran")
SIGNAL_HISTORY_FILE = _get_str("SIGNAL_HISTORY_FILE", "signal_history.csv")

"""
نقطه ورود اصلی ربات.
اجرا با: python -m src.main
هر بار اجرا: طلا + N رمزارز برتر را تحلیل می‌کند، سیگنال‌ها را رتبه‌بندی کرده،
یک ایمیل گزارش می‌فرستد و تاریخچه را برای بک‌تست آینده ثبت می‌کند.
"""
import sys
import time
import traceback
from datetime import datetime

import pytz

from . import config
from .data_fetcher import fetch_crypto_ohlcv, fetch_gold_ohlcv, get_exchange, get_top_crypto_symbols
from .emailer import build_html_report, build_plain_text_report, send_email
from .history import log_signal_history
from .signal_engine import analyze_symbol


def run():
    tz = pytz.timezone(config.TIMEZONE)
    now_str = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[INFO] شروع اجرای ربات: {now_str}")

    results = []

    # --- تحلیل طلا ---
    for ticker in config.GOLD_TICKERS:
        df = fetch_gold_ohlcv(ticker=ticker, interval=config.GOLD_INTERVAL, period=config.GOLD_PERIOD)
        result = analyze_symbol(df, ticker, "GOLD", config.GOLD_INTERVAL)
        if result:
            results.append(result)
            print(f"[GOLD] {ticker}: {result['signal']} (اطمینان={result.get('confidence')})")
        else:
            print(f"[WARN] تحلیل طلا برای {ticker} ممکن نشد (داده ناکافی)")

    # --- تحلیل رمزارزها ---
    exchange = get_exchange()
    try:
        symbols = get_top_crypto_symbols(
            exchange,
            max_symbols=config.MAX_CRYPTO_SYMBOLS,
            min_quote_volume=config.MIN_QUOTE_VOLUME_USDT,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] دریافت لیست نمادهای برتر ناموفق بود: {exc}")
        symbols = []

    print(f"[INFO] تعداد رمزارزهای بررسی‌شونده: {len(symbols)}")

    for symbol in symbols:
        df = fetch_crypto_ohlcv(
            exchange, symbol, timeframe=config.CRYPTO_TIMEFRAME, limit=config.CRYPTO_CANDLE_LIMIT
        )
        result = analyze_symbol(df, symbol, "CRYPTO", config.CRYPTO_TIMEFRAME)
        if result:
            results.append(result)
        time.sleep(max(exchange.rateLimit / 1000, 0.2))

    actionable = [r for r in results if r.get("signal") in ("LONG", "SHORT")]
    actionable.sort(key=lambda r: r["confidence"], reverse=True)

    top_results = list(actionable[: config.TOP_SIGNALS_COUNT])
    gold_symbols_in_top = {r["symbol"] for r in top_results if r["market"] == "GOLD"}
    for r in actionable:
        if r["market"] == "GOLD" and r["symbol"] not in gold_symbols_in_top:
            top_results.append(r)

    print(f"[INFO] تعداد کل سیگنال‌های قابل‌اتکا: {len(actionable)}")

    html = build_html_report(top_results, now_str)
    plain = build_plain_text_report(top_results)

    if config.EMAIL_FROM and config.EMAIL_PASSWORD and config.EMAIL_TO:
        try:
            send_email(
                subject=f"سیگنال بازار کریپتو و طلا - {now_str}",
                html_content=html,
                plain_content=plain,
                smtp_server=config.EMAIL_SMTP_SERVER,
                smtp_port=config.EMAIL_SMTP_PORT,
                sender_email=config.EMAIL_FROM,
                sender_password=config.EMAIL_PASSWORD,
                recipients=config.EMAIL_TO,
            )
            print("[INFO] ایمیل با موفقیت ارسال شد.")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] ارسال ایمیل ناموفق بود: {exc}")
            traceback.print_exc()
    else:
        print("[WARN] تنظیمات ایمیل کامل نیست (EMAIL_FROM/EMAIL_PASSWORD/EMAIL_TO)؛ ایمیلی ارسال نشد.")
        print(plain)

    log_signal_history(results, now_str, path=config.SIGNAL_HISTORY_FILE)
    print("[INFO] اجرای ربات به پایان رسید.")


if __name__ == "__main__":
    try:
        run()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)

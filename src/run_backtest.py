"""
اجرای بک‌تست: python -m src.run_backtest

⚠️ این اسکریپت جدا از ربات زنده (src/main.py) است و نباید در ورک‌فلوی ساعتی
اجرا شود - چون برای هر نماد چند ماه کندل تاریخی می‌گیرد (صدها درخواست شبکه) و
ممکن است چند دقیقه طول بکشد. توصیه می‌شود یک‌بار، به‌صورت دستی (لوکال یا با یک
ورک‌فلوی جدا و workflow_dispatch) اجرا شود.
"""
import pandas as pd

from . import config
from .backtest import simulate_symbol, summarize_trades
from .data_fetcher import fetch_gold_ohlcv, fetch_historical_ohlcv, get_multi_exchange_symbols


def run():
    print("[INFO] شروع بک‌تست...")
    all_trades = []

    # --- بک‌تست فلزات (طلا/نقره) ---
    # یاهو فایننس حداکثر ۶۰ روز کندل ساعتی می‌دهد؛ این محدودیت خود API است.
    for ticker in config.METAL_TICKERS:
        market_label = config.METAL_LABELS.get(ticker, ticker)
        df = fetch_gold_ohlcv(ticker=ticker, interval=config.METAL_INTERVAL, period="60d")
        if df is None or len(df) < 100:
            print(f"[WARN] داده تاریخی کافی برای {ticker} در دسترس نبود")
            continue
        trades = simulate_symbol(
            df, ticker, market_label, config.METAL_INTERVAL,
            max_holding_candles=config.BACKTEST_MAX_HOLDING_CANDLES,
        )
        print(f"[{market_label}] {len(trades)} معامله شبیه‌سازی شد (از {len(df)} کندل)")
        all_trades.extend(trades)

    # --- بک‌تست کریپتو ---
    symbol_exchange_pairs = get_multi_exchange_symbols(
        config.CRYPTO_EXCHANGES,
        max_total_symbols=config.BACKTEST_MAX_SYMBOLS,
        min_quote_volume=config.BACKTEST_MIN_QUOTE_VOLUME_USDT,
        quotes=tuple(config.CRYPTO_QUOTE_CURRENCIES),
        max_exchanges=config.MAX_EXCHANGES_TO_MERGE,
    )
    print(f"[INFO] {len(symbol_exchange_pairs)} رمزارز برای بک‌تست انتخاب شد")

    for symbol, exchange in symbol_exchange_pairs:
        print(f"[INFO] دریافت تاریخچه {symbol} ({config.BACKTEST_DAYS} روز)...")
        df = fetch_historical_ohlcv(exchange, symbol, timeframe=config.CRYPTO_TIMEFRAME, days=config.BACKTEST_DAYS)
        if df is None or len(df) < 100:
            print(f"[WARN] داده تاریخی کافی برای {symbol} نبود")
            continue
        trades = simulate_symbol(
            df, symbol, "CRYPTO", config.CRYPTO_TIMEFRAME,
            max_holding_candles=config.BACKTEST_MAX_HOLDING_CANDLES,
        )
        print(f"[{symbol}] {len(trades)} معامله شبیه‌سازی شد (از {len(df)} کندل)")
        all_trades.extend(trades)

    print(f"\n[INFO] مجموع معاملات شبیه‌سازی‌شده: {len(all_trades)}")

    if not all_trades:
        print("[WARN] هیچ معامله‌ای شبیه‌سازی نشد؛ گزارشی برای نمایش نیست.")
        return

    pd.DataFrame(all_trades).to_csv(config.BACKTEST_OUTPUT_FILE, index=False)
    print(f"[INFO] جزئیات معاملات در {config.BACKTEST_OUTPUT_FILE} ذخیره شد")

    summary = summarize_trades(all_trades)
    print("\n" + "=" * 50)
    print("خلاصه نتایج بک‌تست")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")

    if summary["total_trades"] < 100:
        print(
            f"\n[WARN] تعداد معاملات ({summary['total_trades']}) کمتر از هدف ۱۰۰ است. "
            "برای افزایش، BACKTEST_DAYS یا BACKTEST_MAX_SYMBOLS را در متغیرهای محیطی بیشتر کنید."
        )


if __name__ == "__main__":
    run()

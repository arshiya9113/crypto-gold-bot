"""
ثبت تاریخچه هر سیگنال صادرشده در یک فایل CSV.
این فایل زیرساخت مرحله بعدی پروژه (بک‌تست روی ۱۰۰ موقعیت گذشته) خواهد بود؛
چون برای بک‌تست واقعی باید بدانیم در هر لحظه چه سیگنالی صادر شده تا با قیمت‌های
بعدی مقایسه شود. خود فرآیند بک‌تست در این نسخه پیاده‌سازی نشده است.
"""
import csv
import os

HISTORY_COLUMNS = [
    "run_timestamp",
    "symbol",
    "market",
    "timeframe",
    "signal",
    "confidence",
    "candle_time",
    "evidence_bullish",
    "evidence_bearish",
    "entry",
    "stop_loss",
    "take_profit_1",
    "take_profit_2",
    "risk_reward_1",
    "risk_reward_2",
]


def log_signal_history(results, run_timestamp, path="signal_history.csv"):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for r in results:
            if r.get("signal") not in ("LONG", "SHORT"):
                continue
            writer.writerow({
                "run_timestamp": run_timestamp,
                "symbol": r["symbol"],
                "market": r["market"],
                "timeframe": r["timeframe"],
                "signal": r["signal"],
                "confidence": r["confidence"],
                "candle_time": r.get("candle_time"),
                "evidence_bullish": r.get("evidence_bullish"),
                "evidence_bearish": r.get("evidence_bearish"),
                "entry": r.get("entry"),
                "stop_loss": r.get("stop_loss"),
                "take_profit_1": r.get("take_profit_1"),
                "take_profit_2": r.get("take_profit_2"),
                "risk_reward_1": r.get("risk_reward_1"),
                "risk_reward_2": r.get("risk_reward_2"),
            })

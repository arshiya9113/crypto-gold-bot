"""
موتور بک‌تست: شبیه‌سازی گذشته‌نگر بدون نگاه به آینده (No Look-Ahead).

اصل کار: دقیقا همان تابع analyze_symbol که در اجرای زنده استفاده می‌شود را روی
یک پنجره‌ی رو-به-جلو از داده تاریخی صدا می‌زند (هر بار فقط داده‌ی "تا همین لحظه"
را می‌بیند، نه آینده)، و اگر سیگنالی صادر شود، کندل‌های بعدی را دنبال می‌کند تا
ببیند اول به حد ضرر می‌خورد یا به هدف اول/دوم، دقیقا مثل یک معامله واقعی.

محدودیت‌های شناخته‌شده (برای شفافیت، نه پنهان‌کاری):
- کارمزد صرافی و لغزش قیمت (Slippage) مدل نمی‌شود.
- اگر در یک کندل هم‌زمان SL و TP لمس شوند، به‌صورت محافظه‌کارانه فرض می‌شود
  SL زودتر خورده (بدترین حالت ممکن، نه لزوما واقعیت).
- فقط یک دوره تاریخی مشخص بررسی می‌شود؛ عملکرد در شرایط دیگر بازار می‌تواند
  متفاوت باشد.
- برای هر نماد، معاملات همپوشان مجاز نیستند (بعد از بسته شدن یک معامله، دنبال
  سیگنال بعدی می‌گردد) - شبیه یک معامله‌گر با یک پوزیشن باز در هر زمان.
"""
import pandas as pd

from .signal_engine import MIN_BARS, analyze_symbol


def simulate_symbol(df, symbol, market_type, timeframe, max_holding_candles=48):
    """
    شبیه‌سازی گذشته‌نگر روی یک نماد. خروجی: لیستی از دیکشنری معاملات کامل‌شده.
    """
    trades = []
    n = len(df)
    i = MIN_BARS

    while i < n - 1:
        window = df.iloc[: i + 1].reset_index(drop=True)
        result = analyze_symbol(window, symbol, market_type, timeframe)

        if not result or result.get("signal") not in ("LONG", "SHORT"):
            i += 1
            continue

        signal = result["signal"]
        entry = result["entry"]
        stop_loss = result["stop_loss"]
        take_profit_1 = result["take_profit_1"]
        take_profit_2 = result["take_profit_2"]

        outcome = None
        exit_index = None
        exit_price = None

        last_scan_index = min(i + max_holding_candles, n - 1)
        for j in range(i + 1, last_scan_index + 1):
            high = df["high"].iloc[j]
            low = df["low"].iloc[j]

            if signal == "LONG":
                hit_sl = low <= stop_loss
                hit_tp2 = high >= take_profit_2
                hit_tp1 = high >= take_profit_1
            else:
                hit_sl = high >= stop_loss
                hit_tp2 = low <= take_profit_2
                hit_tp1 = low <= take_profit_1

            if hit_sl:
                outcome, exit_index, exit_price = "SL", j, stop_loss
                break
            if hit_tp2:
                outcome, exit_index, exit_price = "TP2", j, take_profit_2
                break
            if hit_tp1:
                outcome, exit_index, exit_price = "TP1", j, take_profit_1
                break

        if outcome is None:
            outcome = "TIMEOUT"
            exit_index = last_scan_index
            exit_price = df["close"].iloc[exit_index]

        risk = abs(entry - stop_loss)
        if risk > 0:
            if signal == "LONG":
                realized_r = (exit_price - entry) / risk
            else:
                realized_r = (entry - exit_price) / risk
        else:
            realized_r = 0.0

        trades.append({
            "symbol": symbol,
            "market": market_type,
            "signal": signal,
            "confidence": result["confidence"],
            "candle_time": result.get("candle_time"),
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit_1": take_profit_1,
            "take_profit_2": take_profit_2,
            "outcome": outcome,
            "exit_time": df["timestamp"].iloc[exit_index] if "timestamp" in df.columns else None,
            "holding_candles": exit_index - i,
            "realized_r": round(realized_r, 3),
        })

        i = exit_index + 1  # بعد از بسته شدن معامله، دنبال سیگنال بعدی بگرد (بدون هم‌پوشانی)

    return trades


def summarize_trades(trades):
    """آمار خلاصه بک‌تست را از لیست معاملات محاسبه می‌کند."""
    if not trades:
        return {"total_trades": 0}

    df = pd.DataFrame(trades)
    total = len(df)
    wins = df[df["outcome"].isin(["TP1", "TP2"])]
    losses = df[df["outcome"] == "SL"]
    timeouts = df[df["outcome"] == "TIMEOUT"]

    win_rate = round(len(wins) / total * 100, 1) if total else 0.0
    avg_r = round(df["realized_r"].mean(), 3) if total else 0.0
    expectancy = avg_r  # میانگین R به‌ازای هر معامله = انتظار ریاضی به واحد ریسک

    summary = {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "timeouts": len(timeouts),
        "win_rate_pct": win_rate,
        "avg_realized_r": avg_r,
        "expectancy_r": expectancy,
    }

    for market in df["market"].unique():
        sub = df[df["market"] == market]
        sub_wins = sub[sub["outcome"].isin(["TP1", "TP2"])]
        summary[f"win_rate_{market}"] = round(len(sub_wins) / len(sub) * 100, 1) if len(sub) else 0.0
        summary[f"trades_{market}"] = len(sub)

    strong = df[df["confidence"] >= 55]
    moderate = df[df["confidence"] < 55]
    for label, sub in [("STRONG", strong), ("MODERATE", moderate)]:
        if len(sub):
            sub_wins = sub[sub["outcome"].isin(["TP1", "TP2"])]
            summary[f"win_rate_{label}"] = round(len(sub_wins) / len(sub) * 100, 1)
            summary[f"trades_{label}"] = len(sub)

    return summary

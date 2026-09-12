"""
موتور تصمیم‌گیری: خروجی اندیکاتورهای تکنیکال و پرایس‌اکشن را با یک سیستم امتیازدهی
وزن‌دار ترکیب می‌کند و برای هر نماد یک سیگنال LONG / SHORT / NEUTRAL همراه با
قیمت ورود، حد ضرر، دو هدف سود و دلایل متنی سیگنال تولید می‌کند.

سیستم امتیاز از -100 (شورت بسیار قوی) تا +100 (لانگ بسیار قوی) است.
"""
import numpy as np

from .indicators import compute_indicators
from .price_action import (
    detect_accumulation_signal,
    detect_candlestick_pattern,
    detect_star_patterns,
    detect_trend_structure,
    detect_volatility_squeeze,
    donchian_breakout,
    find_swing_points,
    nearest_levels,
)

MIN_BARS = 60
STRONG_THRESHOLD = 55
MODERATE_THRESHOLD = 25


def analyze_symbol(df, symbol, market_type, timeframe):
    """تحلیل کامل یک نماد و بازگرداندن دیکشنری نتیجه، یا None اگر داده کافی نبود."""
    if df is None or len(df) < MIN_BARS:
        return None

    df = compute_indicators(df)
    df = df.dropna(subset=["rsi14", "macd_diff", "atr14"]).reset_index(drop=True)
    if len(df) < MIN_BARS:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = float(last["close"])

    score = 0.0
    reasons = []

    # --- ۱. تحلیل روند با میانگین‌های متحرک ---
    if last["ema9"] > last["ema21"] > last["ema50"]:
        score += 20
        reasons.append("روند صعودی: EMA9 بالای EMA21 و EMA21 بالای EMA50")
    elif last["ema9"] < last["ema21"] < last["ema50"]:
        score -= 20
        reasons.append("روند نزولی: EMA9 زیر EMA21 و EMA21 زیر EMA50")

    if not np.isnan(last.get("ema200", np.nan)):
        if close > last["ema200"]:
            score += 10
            reasons.append("قیمت بالای EMA200 (تایید روند بلندمدت صعودی)")
        else:
            score -= 10
            reasons.append("قیمت زیر EMA200 (تایید روند بلندمدت نزولی)")

    # --- ۲. قدرت روند با ADX/DI ---
    if not np.isnan(last.get("adx", np.nan)):
        if last["adx"] > 25:
            if last["plus_di"] > last["minus_di"]:
                score += 10
                reasons.append(f"روند قوی و صعودی (ADX={last['adx']:.1f})")
            else:
                score -= 10
                reasons.append(f"روند قوی و نزولی (ADX={last['adx']:.1f})")
        else:
            reasons.append(f"روند ضعیف / بازار رنج (ADX={last['adx']:.1f})")

    # --- ۳. مومنتوم با MACD ---
    if last["macd_diff"] > 0 and last["macd_diff"] >= prev["macd_diff"]:
        score += 10
        reasons.append("هیستوگرام MACD مثبت و در حال افزایش")
    elif last["macd_diff"] < 0 and last["macd_diff"] <= prev["macd_diff"]:
        score -= 10
        reasons.append("هیستوگرام MACD منفی و در حال کاهش")

    # --- ۴. RSI ---
    rsi = last["rsi14"]
    if 50 < rsi < 70:
        score += 8
        reasons.append(f"RSI در محدوده مومنتوم صعودی سالم ({rsi:.1f})")
    elif rsi >= 70:
        score += 2
        reasons.append(f"RSI در ناحیه اشباع خرید ({rsi:.1f}) - احتیاط برای ورود جدید لانگ")
    elif 30 < rsi < 50:
        score -= 8
        reasons.append(f"RSI در محدوده مومنتوم نزولی ({rsi:.1f})")
    else:
        score -= 2
        reasons.append(f"RSI در ناحیه اشباع فروش ({rsi:.1f}) - احتیاط برای ورود جدید شورت")

    # --- ۵. تقاطع Stochastic RSI ---
    k, d = last.get("stoch_rsi_k"), last.get("stoch_rsi_d")
    pk, pd_prev = prev.get("stoch_rsi_k"), prev.get("stoch_rsi_d")
    if not any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [k, d, pk, pd_prev]):
        if pk < pd_prev and k > d and k < 0.3:
            score += 6
            reasons.append("تقاطع صعودی Stochastic RSI از ناحیه اشباع فروش")
        elif pk > pd_prev and k < d and k > 0.7:
            score -= 6
            reasons.append("تقاطع نزولی Stochastic RSI از ناحیه اشباع خرید")

    # --- ۶. باندهای بولینگر ---
    if close >= last["bb_high"]:
        score += 5
        reasons.append("قیمت در باند بالایی بولینگر (فشار خریدار)")
    elif close <= last["bb_low"]:
        score -= 5
        reasons.append("قیمت در باند پایینی بولینگر (فشار فروشنده)")

    # --- ۷. شکست کانال دونچیان ---
    breakout_dir, breakout_score = donchian_breakout(df, period=20)
    if breakout_dir != "NONE":
        score += breakout_score
        direction_fa = "بالا" if breakout_dir == "UP" else "پایین"
        reasons.append(f"شکست کانال دونچیان ۲۰ کندلی به سمت {direction_fa}")

    # --- ۸. حجم معاملات ---
    vol_sma20 = last.get("vol_sma20")
    if vol_sma20 and not np.isnan(vol_sma20) and vol_sma20 > 0:
        if last["volume"] > 1.5 * vol_sma20:
            if score > 0:
                score += 5
                reasons.append("افزایش قابل توجه حجم معاملات هم‌راستا با سیگنال صعودی")
            elif score < 0:
                score -= 5
                reasons.append("افزایش قابل توجه حجم معاملات هم‌راستا با سیگنال نزولی")

    # --- ۹. ساختار پرایس‌اکشن (سوینگ‌ها) ---
    swing_highs, swing_lows = find_swing_points(df, order=3)
    structure, structure_score = detect_trend_structure(swing_highs, swing_lows)
    score += structure_score
    if structure == "UPTREND":
        reasons.append("ساختار قیمتی صعودی: سقف و کف‌های بالاتر (HH/HL)")
    elif structure == "DOWNTREND":
        reasons.append("ساختار قیمتی نزولی: سقف و کف‌های پایین‌تر (LH/LL)")

    # --- ۱۰. الگوهای کندل استیک ---
    for name, points in detect_candlestick_pattern(df) + detect_star_patterns(df):
        score += points
        if points != 0:
            reasons.append(f"الگوی کندلی: {name}")

    # --- ۱۱. رصد قبل از پامپ: فشردگی نوسان و انباشت مشکوک ---
    watch_flags = []
    squeeze_state = detect_volatility_squeeze(df, lookback=6)
    if squeeze_state == "SQUEEZE_ON":
        watch_flags.append("فشردگی نوسان فعال (Squeeze) - نوسان به‌شدت پایین آمده، احتمال حرکت انفجاری در کندل‌های آینده")
    elif squeeze_state == "SQUEEZE_RELEASED":
        watch_flags.append("فشردگی نوسان به‌تازگی آزاد شده - احتمال شروع حرکت شارپ در همین لحظه")

    if detect_accumulation_signal(df, lookback=5):
        watch_flags.append("حجم معاملات غیرعادی بالا همراه با نوسان قیمت بسیار کم - نشانه احتمالی انباشت/توزیع سنگین")

    score = max(-100.0, min(100.0, score))
    confidence = round(abs(score), 1)

    if score >= STRONG_THRESHOLD:
        signal = "LONG"
    elif score <= -STRONG_THRESHOLD:
        signal = "SHORT"
    elif score >= MODERATE_THRESHOLD:
        signal = "LONG"
        reasons.insert(0, "سیگنال با اطمینان متوسط (نه بسیار قوی)")
    elif score <= -MODERATE_THRESHOLD:
        signal = "SHORT"
        reasons.insert(0, "سیگنال با اطمینان متوسط (نه بسیار قوی)")
    else:
        signal = "NEUTRAL"

    if signal == "NEUTRAL":
        return {
            "symbol": symbol,
            "market": market_type,
            "timeframe": timeframe,
            "signal": signal,
            "confidence": confidence,
            "price": close,
            "watch_flags": watch_flags,
        }

    atr = float(last["atr14"])
    support, resistance = nearest_levels(swing_highs, swing_lows, close)

    if signal == "LONG":
        sl_by_atr = close - 1.5 * atr
        sl_by_structure = support * 0.998 if support else sl_by_atr
        stop_loss = max(min(sl_by_atr, sl_by_structure), close - 4 * atr)
        risk = close - stop_loss
        if risk <= 0:
            return None
        take_profit_1 = close + risk * 2
        take_profit_2 = close + risk * 3
        if resistance and resistance < take_profit_1:
            take_profit_1 = resistance
    else:  # SHORT
        sl_by_atr = close + 1.5 * atr
        sl_by_structure = resistance * 1.002 if resistance else sl_by_atr
        stop_loss = min(max(sl_by_atr, sl_by_structure), close + 4 * atr)
        risk = stop_loss - close
        if risk <= 0:
            return None
        take_profit_1 = close - risk * 2
        take_profit_2 = close - risk * 3
        if support and support > take_profit_1:
            take_profit_1 = support

    risk_reward_1 = round(abs(take_profit_1 - close) / risk, 2)
    risk_reward_2 = round(abs(take_profit_2 - close) / risk, 2)

    return {
        "symbol": symbol,
        "market": market_type,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "price": close,
        "entry": close,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk_reward_1": risk_reward_1,
        "risk_reward_2": risk_reward_2,
        "atr": atr,
        "reasons": reasons,
        "watch_flags": watch_flags,
    }

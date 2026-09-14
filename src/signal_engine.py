"""
موتور تصمیم‌گیری - نسخه پرایس‌اکشن‌محور.

فلسفه طراحی: مسیر اصلی تحلیل، پرایس‌اکشن است (ساختار بازار، Order Block،
Fair Value Gap، الگوهای کلاسیک چارتی مثل سر و شانه/دوقلو/مثلث/کانال، کندل‌های
اسپایک، و اندیکاتور اختصاصی SPI). اندیکاتورهای تکنیکال کلاسیک (EMA/RSI/MACD/...)
حذف نشده‌اند، ولی نقششان "تاییدیه" روی تز اصلی پرایس‌اکشن است، نه محرک اصلی امتیاز.

سیستم امتیاز از -100 (شورت بسیار قوی) تا +100 (لانگ بسیار قوی) است. برای شفافیت
و کاهش حس "تناقض" در سیگنال‌ها، مجموع شواهد صعودی و نزولی جداگانه هم محاسبه و در
خروجی گزارش می‌شود (evidence_bullish / evidence_bearish) تا مشخص باشد تصمیم نهایی
حاصل چه میزان کشمکش بین شواهد موافق و مخالف بوده است.

نکته مهم درباره پایداری سیگنال: این تابع فرض می‌کند df فقط شامل کندل‌های کاملا
بسته‌شده است (کندل در-حال-شکل‌گیری قبلا در data_fetcher حذف شده). این فرض حیاتی
است چون تحلیل روی کندل ناقص باعث سیگنال‌های ناپایدار/متناقض و ATR مصنوعا کوچک
(در نتیجه SL/TP غیرواقعی و زود لمس‌شونده) می‌شود.

هر نتیجه همچنین می‌تواند "watch_flags" داشته باشد: هشدارهای غیرجهت‌دار درباره
احتمال یک حرکت بزرگ نزدیک (فشردگی نوسان، انباشت حجم، الگوهای در حال شکل‌گیری).
"""
import numpy as np

from .chart_patterns import (
    channel_position,
    detect_channel,
    detect_climax_spike,
    detect_double_bottom,
    detect_double_top,
    detect_head_and_shoulders,
    detect_inverse_head_and_shoulders,
    detect_triangle,
    estimate_elliott_wave_hint,
)
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
from .smart_pressure_index import compute_spi, detect_spi_divergence, detect_spi_exhaustion
from .smc_patterns import (
    breaker_block_reaction,
    compute_htf_bias,
    compute_premium_discount_zone,
    detect_breaker_block,
    detect_equal_highs_lows,
    detect_fvg,
    detect_liquidity_sweep,
    detect_order_blocks,
    nearest_fvg_reaction,
    order_block_reaction,
    price_zone,
)

MIN_BARS = 90
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
    atr = float(last["atr14"])
    candle_time = last["timestamp"] if "timestamp" in df.columns else None

    reasons_primary = []    # دلایل پرایس‌اکشنی (تز اصلی)
    reasons_secondary = []  # تاییدیه‌های اندیکاتوری (فرعی)
    watch_flags = []        # هشدارهای غیرجهت‌دار (احتمال حرکت بزرگ نزدیک)

    state = {"score": 0.0, "bull": 0.0, "bear": 0.0}

    def add(points, text=None, primary=True):
        """امتیاز را اعمال می‌کند، مجموع شواهد له/علیه را به‌روزرسانی می‌کند و در
        صورت وجود متن، آن را به لیست دلایل مناسب اضافه می‌کند."""
        state["score"] += points
        if points > 0:
            state["bull"] += points
        elif points < 0:
            state["bear"] += abs(points)
        if text:
            (reasons_primary if primary else reasons_secondary).append(text)

    swing_highs, swing_lows = find_swing_points(df, order=3)

    # =========================================================================
    # بخش اول (اصلی): پرایس‌اکشن
    # =========================================================================

    # --- ۱. ساختار بازار و شکست ساختار (BOS در برابر CHoCH) ---
    structure, structure_score = detect_trend_structure(swing_highs, swing_lows)
    if structure == "UPTREND":
        add(structure_score, "ساختار قیمتی صعودی: سقف و کف‌های بالاتر (HH/HL)")
    elif structure == "DOWNTREND":
        add(structure_score, "ساختار قیمتی نزولی: سقف و کف‌های پایین‌تر (LH/LL)")

    last_swing_high = swing_highs[-1][1] if swing_highs else None
    last_swing_low = swing_lows[-1][1] if swing_lows else None
    if last_swing_high and close > last_swing_high:
        if structure == "DOWNTREND":
            add(20, f"تغییر کاراکتر بازار (CHoCH) به سمت صعودی: قیمت بالای آخرین سقف سوینگ "
                    f"({last_swing_high:.4g}) در یک روند نزولی قبلی بسته شده - هشدار احتمال برگشت روند")
        else:
            add(15, f"شکست ساختار به سمت بالا (BOS): قیمت بالای آخرین سقف سوینگ ({last_swing_high:.4g}) بسته شده")
    if last_swing_low and close < last_swing_low:
        if structure == "UPTREND":
            add(-20, f"تغییر کاراکتر بازار (CHoCH) به سمت نزولی: قیمت زیر آخرین کف سوینگ "
                     f"({last_swing_low:.4g}) در یک روند صعودی قبلی بسته شده - هشدار احتمال برگشت روند")
        else:
            add(-15, f"شکست ساختار به سمت پایین (BOS): قیمت زیر آخرین کف سوینگ ({last_swing_low:.4g}) بسته شده")

    # --- ۲. Order Block (ناحیه احتمالی ورود پول بزرگ) ---
    bullish_ob, bearish_ob = detect_order_blocks(df, swing_highs, swing_lows, lookback=80)
    ob_reaction = order_block_reaction(bullish_ob, bearish_ob, close, atr)
    if ob_reaction == "BULLISH_OB_REACTION":
        add(15, "قیمت به یک Order Block صعودی بازگشته (ناحیه احتمالی ورود خریداران بزرگ)")
    elif ob_reaction == "BEARISH_OB_REACTION":
        add(-15, "قیمت به یک Order Block نزولی بازگشته (ناحیه احتمالی ورود فروشندگان بزرگ)")

    # --- ۳. Fair Value Gap (FVG) ---
    bullish_fvg, bearish_fvg = detect_fvg(df, lookback=60)
    fvg_reaction = nearest_fvg_reaction(bullish_fvg, bearish_fvg, close, atr)
    if fvg_reaction == "BULLISH_FVG_REACTION":
        add(10, "قیمت در ناحیه Fair Value Gap صعودی پرنشده قرار دارد (احتمال واکنش خریداران)")
    elif fvg_reaction == "BEARISH_FVG_REACTION":
        add(-10, "قیمت در ناحیه Fair Value Gap نزولی پرنشده قرار دارد (احتمال واکنش فروشندگان)")

    # --- ۳.۱ بریکر بلاک (Order Block شکسته‌شده و معکوس‌شده) ---
    breaker = detect_breaker_block(df, bullish_ob, bearish_ob)
    breaker_reaction = breaker_block_reaction(breaker, close, atr)
    if breaker_reaction == "BULLISH_BREAKER_REACTION":
        add(14, "قیمت به یک Bullish Breaker Block بازگشته (ناحیه‌ای که قبلا مقاومت بوده و اکنون حمایت است)")
    elif breaker_reaction == "BEARISH_BREAKER_REACTION":
        add(-14, "قیمت به یک Bearish Breaker Block بازگشته (ناحیه‌ای که قبلا حمایت بوده و اکنون مقاومت است)")

    # --- ۳.۲ شکار نقدینگی / استاپ‌هانت ---
    liquidity_sweep = detect_liquidity_sweep(df, swing_highs, swing_lows)
    if liquidity_sweep == "BULLISH_LIQUIDITY_SWEEP":
        add(16, "شکار نقدینگی کف (Liquidity Sweep): سایه کندل کف سوینگ را جارو کرده ولی بسته شدن بالای آن - نشانه قوی برگشت صعودی")
    elif liquidity_sweep == "BEARISH_LIQUIDITY_SWEEP":
        add(-16, "شکار نقدینگی سقف (Liquidity Sweep): سایه کندل سقف سوینگ را جارو کرده ولی بسته شدن پایین آن - نشانه قوی برگشت نزولی")

    # --- ۳.۳ سقف/کف‌های برابر (نواحی تجمع نقدینگی) - فقط هشدار رصد، بدون امتیاز ---
    equal_levels = detect_equal_highs_lows(swing_highs, swing_lows)
    if equal_levels["equal_highs"] and close < equal_levels["equal_highs"]:
        if (equal_levels["equal_highs"] - close) / close < 0.01:
            watch_flags.append(
                f"قیمت نزدیک سقف‌های برابر (نقدینگی خرید در {equal_levels['equal_highs']:.4g}) - "
                "احتمال جارو شدن این ناحیه قبل از حرکت اصلی"
            )
    if equal_levels["equal_lows"] and close > equal_levels["equal_lows"]:
        if (close - equal_levels["equal_lows"]) / close < 0.01:
            watch_flags.append(
                f"قیمت نزدیک کف‌های برابر (نقدینگی فروش در {equal_levels['equal_lows']:.4g}) - "
                "احتمال جارو شدن این ناحیه قبل از حرکت اصلی"
            )

    # --- ۳.۴ ناحیه پرمیوم/دیسکانت (تعادل لگ اخیر) ---
    pd_zone = compute_premium_discount_zone(swing_highs, swing_lows)
    zone_label = price_zone(pd_zone, close)
    if zone_label == "DISCOUNT":
        if state["score"] > 0:
            add(6, "قیمت در ناحیه دیسکانت (Discount) لگ اخیر - هم‌راستا با تز خرید طبق مفهوم پرمیوم/دیسکانت")
        elif state["score"] < 0:
            add(4, "هشدار پرمیوم/دیسکانت: قیمت در ناحیه دیسکانت است؛ فروش از این نقطه خلاف اصول پول هوشمند است")
    elif zone_label == "PREMIUM":
        if state["score"] < 0:
            add(-6, "قیمت در ناحیه پرمیوم (Premium) لگ اخیر - هم‌راستا با تز فروش طبق مفهوم پرمیوم/دیسکانت")
        elif state["score"] > 0:
            add(-4, "هشدار پرمیوم/دیسکانت: قیمت در ناحیه پرمیوم است؛ خرید از این نقطه خلاف اصول پول هوشمند است")

    # --- ۳.۵ شاخص فشار هوشمند (Smart Pressure Index - اندیکاتور اختصاصی) ---
    spi_raw, spi_smoothed, spi_accumulated = compute_spi(df)
    spi_divergence = detect_spi_divergence(swing_highs, swing_lows, spi_accumulated)
    if spi_divergence == "BULLISH_DIVERGENCE":
        add(18, "اندیکاتور اختصاصی SPI: واگرایی صعودی - قیمت کف پایین‌تر زده ولی فشار انباشتی SPI "
                "(ترکیب بدنه/سایه/حجم) ضعف کمتری نشان می‌دهد؛ نشانه احتمالی ضعف فروشندگان")
    elif spi_divergence == "BEARISH_DIVERGENCE":
        add(-18, "اندیکاتور اختصاصی SPI: واگرایی نزولی - قیمت سقف بالاتر زده ولی فشار انباشتی SPI "
                 "ضعف بیشتری نشان می‌دهد؛ نشانه احتمالی ضعف خریداران")

    spi_exhaustion = detect_spi_exhaustion(spi_smoothed)
    if spi_exhaustion == "OVERHEATED_BUY_PRESSURE":
        add(-8, "اندیکاتور اختصاصی SPI: فشار خرید به‌طور غیرعادی بالا رفته - احتمال اصلاح کوتاه‌مدت")
    elif spi_exhaustion == "OVERHEATED_SELL_PRESSURE":
        add(8, "اندیکاتور اختصاصی SPI: فشار فروش به‌طور غیرعادی بالا رفته - احتمال اصلاح کوتاه‌مدت")

    if not spi_smoothed.empty and not np.isnan(spi_smoothed.iloc[-1]):
        if spi_smoothed.iloc[-1] > 0.15:
            add(5, "اندیکاتور اختصاصی SPI: فشار لحظه‌ای خریداران (بدنه+سایه+حجم) مثبت است")
        elif spi_smoothed.iloc[-1] < -0.15:
            add(-5, "اندیکاتور اختصاصی SPI: فشار لحظه‌ای فروشندگان (بدنه+سایه+حجم) منفی است")

    # --- ۴. الگوهای بازگشتی کلاسیک (سر و شانه / دوقلو) ---
    hs = detect_head_and_shoulders(swing_highs, swing_lows)
    if hs:
        if close < hs["neckline"]:
            add(-22, "الگوی سر و شانه تکمیل شده و نکلاین شکسته شده (سیگنال نزولی قوی)")
        else:
            add(-5)
            watch_flags.append("الگوی سر و شانه در حال شکل‌گیری - در صورت شکست نکلاین، سیگنال نزولی تایید می‌شود")

    inv_hs = detect_inverse_head_and_shoulders(swing_highs, swing_lows)
    if inv_hs:
        if close > inv_hs["neckline"]:
            add(22, "الگوی سر و شانه معکوس تکمیل شده و نکلاین شکسته شده (سیگنال صعودی قوی)")
        else:
            add(5)
            watch_flags.append("الگوی سر و شانه معکوس در حال شکل‌گیری - در صورت شکست نکلاین، سیگنال صعودی تایید می‌شود")

    double_top = detect_double_top(swing_highs, swing_lows)
    if double_top:
        if close < double_top["neckline"]:
            add(-18, "الگوی سقف دوقلو تکمیل شده و نکلاین شکسته شده (سیگنال نزولی)")
        else:
            add(-4)
            watch_flags.append("سقف دوقلو در حال شکل‌گیری - منتظر شکست نکلاین برای تایید")

    double_bottom = detect_double_bottom(swing_highs, swing_lows)
    if double_bottom:
        if close > double_bottom["neckline"]:
            add(18, "الگوی کف دوقلو تکمیل شده و نکلاین شکسته شده (سیگنال صعودی)")
        else:
            add(4)
            watch_flags.append("کف دوقلو در حال شکل‌گیری - منتظر شکست نکلاین برای تایید")

    # --- ۵. مثلث ---
    triangle = detect_triangle(swing_highs, swing_lows)
    if triangle == "ASCENDING":
        add(8, "مثلث صعودی در حال شکل‌گیری (تمایل آماری به شکست رو به بالا)")
    elif triangle == "DESCENDING":
        add(-8, "مثلث نزولی در حال شکل‌گیری (تمایل آماری به شکست رو به پایین)")
    elif triangle == "SYMMETRICAL":
        watch_flags.append("مثلث متقارن در حال شکل‌گیری - جهت شکست هنوز مشخص نیست، آماده‌باش برای هر دو سمت")

    # --- ۶. کانال قیمتی ---
    channel = detect_channel(swing_highs, swing_lows)
    if channel:
        pos = channel_position(channel, close)
        if pos is not None:
            if channel["type"] == "ASCENDING":
                if pos < 0.25:
                    add(8, "قیمت نزدیک کف یک کانال صعودی - منطقه احتمالی خرید در جهت روند")
                elif pos > 0.85:
                    add(-4, "قیمت نزدیک سقف یک کانال صعودی - احتیاط، ریسک اصلاح کوتاه‌مدت")
            elif channel["type"] == "DESCENDING":
                if pos > 0.75:
                    add(-8, "قیمت نزدیک سقف یک کانال نزولی - منطقه احتمالی فروش در جهت روند")
                elif pos < 0.15:
                    add(4, "قیمت نزدیک کف یک کانال نزولی - احتمال برگشت موقت")
            else:  # HORIZONTAL
                if pos < 0.2:
                    add(6, "قیمت نزدیک کف محدوده رنج - منطقه احتمالی خرید")
                elif pos > 0.8:
                    add(-6, "قیمت نزدیک سقف محدوده رنج - منطقه احتمالی فروش")

    # --- ۷. کندل اسپایک/کلایمکس ---
    spike = detect_climax_spike(df)
    if spike == "BUY_CLIMAX":
        add(-12, "کندل اسپایک صعودی با رد قیمت از بالا (Buying Climax) - هشدار احتمال برگشت نزولی")
    elif spike == "SELL_CLIMAX":
        add(12, "کندل اسپایک نزولی با رد قیمت از پایین (Selling Climax) - هشدار احتمال برگشت صعودی")
    elif spike == "MOMENTUM_SPIKE_UP":
        add(6, "کندل مومنتومی قوی صعودی (بدون رد شدن قیمت)")
    elif spike == "MOMENTUM_SPIKE_DOWN":
        add(-6, "کندل مومنتومی قوی نزولی (بدون رد شدن قیمت)")

    # --- ۸. الگوهای کندل استیک ---
    for name, points in detect_candlestick_pattern(df) + detect_star_patterns(df):
        if points != 0:
            add(points, f"الگوی کندلی: {name}")

    # --- ۹. شکست کانال دونچیان (تاییدیه شکست، وزن کمتر از BOS ساختاری) ---
    breakout_dir, breakout_score = donchian_breakout(df, period=20)
    if breakout_dir != "NONE":
        direction_fa = "بالا" if breakout_dir == "UP" else "پایین"
        add(breakout_score * 0.55, f"شکست کانال دونچیان ۲۰ کندلی به سمت {direction_fa}")

    # --- ۱۰. تخمین بسیار اکتشافی موج الیوت (وزن بسیار کم و کاملا غیرقطعی) ---
    elliott_hint = estimate_elliott_wave_hint(swing_highs, swing_lows)
    if elliott_hint == "IMPULSE_UP":
        add(3, "تخمین اکتشافی: احتمال قرارگیری در یک موج صعودی ایمپالسیو "
               "(⚠️ تخمین ساده بر پایه تناوب سوینگ‌ها، نه شمارش دقیق الیوت)")
    elif elliott_hint == "IMPULSE_DOWN":
        add(-3, "تخمین اکتشافی: احتمال قرارگیری در یک موج نزولی ایمپالسیو "
                "(⚠️ تخمین ساده بر پایه تناوب سوینگ‌ها، نه شمارش دقیق الیوت)")

    # --- ۱۱. تاییدیه تحلیل بالا-به-پایین (Top-Down): جهت‌گیری تایم‌فریم بالاتر ---
    htf_bias = compute_htf_bias(df, rule="4h")
    if htf_bias == "BULLISH":
        if state["score"] > 0:
            add(5, "تاییدیه تایم‌فریم بالاتر (۴ ساعته): روند کلی هم‌جهت صعودی است")
        elif state["score"] < 0:
            add(3, "هشدار: روند تایم‌فریم بالاتر (۴ ساعته) صعودی است؛ در تضاد با تز فروش فعلی")
    elif htf_bias == "BEARISH":
        if state["score"] < 0:
            add(-5, "تاییدیه تایم‌فریم بالاتر (۴ ساعته): روند کلی هم‌جهت نزولی است")
        elif state["score"] > 0:
            add(-3, "هشدار: روند تایم‌فریم بالاتر (۴ ساعته) نزولی است؛ در تضاد با تز خرید فعلی")

    # =========================================================================
    # بخش دوم (فرعی/تاییدیه): اندیکاتورهای تکنیکال
    # =========================================================================

    if last["ema9"] > last["ema21"] > last["ema50"]:
        add(8, "تاییدیه اندیکاتوری: تراز صعودی EMA9 > EMA21 > EMA50", primary=False)
    elif last["ema9"] < last["ema21"] < last["ema50"]:
        add(-8, "تاییدیه اندیکاتوری: تراز نزولی EMA9 < EMA21 < EMA50", primary=False)

    if not np.isnan(last.get("ema200", np.nan)):
        if close > last["ema200"]:
            add(5, "تاییدیه اندیکاتوری: قیمت بالای EMA200 (روند بلندمدت صعودی)", primary=False)
        else:
            add(-5, "تاییدیه اندیکاتوری: قیمت زیر EMA200 (روند بلندمدت نزولی)", primary=False)

    if not np.isnan(last.get("adx", np.nan)):
        if last["adx"] > 25:
            if last["plus_di"] > last["minus_di"]:
                add(6, f"تاییدیه اندیکاتوری: روند قوی و صعودی (ADX={last['adx']:.1f})", primary=False)
            else:
                add(-6, f"تاییدیه اندیکاتوری: روند قوی و نزولی (ADX={last['adx']:.1f})", primary=False)

    if last["macd_diff"] > 0 and last["macd_diff"] >= prev["macd_diff"]:
        add(6, "تاییدیه اندیکاتوری: هیستوگرام MACD مثبت و در حال افزایش", primary=False)
    elif last["macd_diff"] < 0 and last["macd_diff"] <= prev["macd_diff"]:
        add(-6, "تاییدیه اندیکاتوری: هیستوگرام MACD منفی و در حال کاهش", primary=False)

    rsi = last["rsi14"]
    if 50 < rsi < 70:
        add(5, f"تاییدیه اندیکاتوری: RSI در محدوده مومنتوم صعودی سالم ({rsi:.1f})", primary=False)
    elif rsi >= 70:
        add(1, f"تاییدیه اندیکاتوری: RSI اشباع خرید ({rsi:.1f}) - احتیاط", primary=False)
    elif 30 < rsi < 50:
        add(-5, f"تاییدیه اندیکاتوری: RSI در محدوده مومنتوم نزولی ({rsi:.1f})", primary=False)
    else:
        add(-1, f"تاییدیه اندیکاتوری: RSI اشباع فروش ({rsi:.1f}) - احتیاط", primary=False)

    k, d = last.get("stoch_rsi_k"), last.get("stoch_rsi_d")
    pk, pd_prev = prev.get("stoch_rsi_k"), prev.get("stoch_rsi_d")
    if not any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [k, d, pk, pd_prev]):
        if pk < pd_prev and k > d and k < 0.3:
            add(4, "تاییدیه اندیکاتوری: تقاطع صعودی Stochastic RSI از ناحیه اشباع فروش", primary=False)
        elif pk > pd_prev and k < d and k > 0.7:
            add(-4, "تاییدیه اندیکاتوری: تقاطع نزولی Stochastic RSI از ناحیه اشباع خرید", primary=False)

    if close >= last["bb_high"]:
        add(3, "تاییدیه اندیکاتوری: قیمت در باند بالایی بولینگر", primary=False)
    elif close <= last["bb_low"]:
        add(-3, "تاییدیه اندیکاتوری: قیمت در باند پایینی بولینگر", primary=False)

    vol_sma20 = last.get("vol_sma20")
    if vol_sma20 and not np.isnan(vol_sma20) and vol_sma20 > 0 and last["volume"] > 1.5 * vol_sma20:
        if state["score"] > 0:
            add(4, "تاییدیه اندیکاتوری: افزایش حجم معاملات هم‌راستا با تز صعودی", primary=False)
        elif state["score"] < 0:
            add(-4, "تاییدیه اندیکاتوری: افزایش حجم معاملات هم‌راستا با تز نزولی", primary=False)

    # =========================================================================
    # بخش سوم: رصد قبل از پامپ (فشردگی نوسان و انباشت مشکوک)
    # =========================================================================
    squeeze_state = detect_volatility_squeeze(df, lookback=6)
    if squeeze_state == "SQUEEZE_ON":
        watch_flags.append("فشردگی نوسان فعال (Squeeze) - نوسان به‌شدت پایین آمده، احتمال حرکت انفجاری در کندل‌های آینده")
    elif squeeze_state == "SQUEEZE_RELEASED":
        watch_flags.append("فشردگی نوسان به‌تازگی آزاد شده - احتمال شروع حرکت شارپ در همین لحظه")

    if detect_accumulation_signal(df, lookback=5):
        watch_flags.append("حجم معاملات غیرعادی بالا همراه با نوسان قیمت بسیار کم - نشانه احتمالی انباشت/توزیع سنگین")

    # =========================================================================
    # جمع‌بندی و تصمیم نهایی
    # =========================================================================
    score = max(-100.0, min(100.0, state["score"]))
    confidence = round(abs(score), 1)
    evidence_bullish = round(state["bull"], 1)
    evidence_bearish = round(state["bear"], 1)
    reasons = reasons_primary + (["— تاییدیه‌های اندیکاتوری —"] if reasons_secondary else []) + reasons_secondary

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
            "candle_time": candle_time,
            "evidence_bullish": evidence_bullish,
            "evidence_bearish": evidence_bearish,
            "watch_flags": watch_flags,
        }

    support, resistance = nearest_levels(swing_highs, swing_lows, close)
    MIN_ACCEPTABLE_RR = 1.2  # اگر حمایت/مقاومت خیلی نزدیک باشد، TP1 را با آن جایگزین نکن

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
            candidate_rr = (resistance - close) / risk
            if candidate_rr >= MIN_ACCEPTABLE_RR:
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
            candidate_rr = (close - support) / risk
            if candidate_rr >= MIN_ACCEPTABLE_RR:
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
        "candle_time": candle_time,
        "evidence_bullish": evidence_bullish,
        "evidence_bearish": evidence_bearish,
        "stop_loss": stop_loss,
        "take_profit_1": take_profit_1,
        "take_profit_2": take_profit_2,
        "risk_reward_1": risk_reward_1,
        "risk_reward_2": risk_reward_2,
        "atr": atr,
        "reasons": reasons,
        "watch_flags": watch_flags,
    }

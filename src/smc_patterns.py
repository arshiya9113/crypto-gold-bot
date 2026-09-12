"""
مفاهیم پرایس‌اکشن مکتب Smart Money / ICT:
- Fair Value Gap (FVG): ناحیه‌ای که در آن قیمت با شتاب حرکت کرده و یک "خلأ" بین
  سایه کندل اول و سوم (در یک الگوی سه‌کندلی) باقی گذاشته. این نواحی اغلب بعدا
  به‌عنوان منطقه حمایت/مقاومت (Magnet) توسط قیمت بازدید می‌شوند.
- Order Block: آخرین کندل مخالفِ جهتِ حرکت، درست قبل از یک کندل جابه‌جایی
  (Displacement) قدرتمند که ساختار قیمتی قبلی را می‌شکند. این نواحی اغلب به‌عنوان
  منطقه احتمالی ورود سرمایه بزرگ در نظر گرفته می‌شوند.

نکته صادقانه: این دو مفهوم در تحلیل «پول هوشمند» بسیار پرطرفدارند ولی قطعیت
علمی/آماری اثبات‌شده‌ای مثل یک قانون فیزیکی ندارند؛ اینجا با تعریف‌های متداول و
قابل محاسبه از جامعه معامله‌گری پیاده‌سازی شده‌اند.
"""


def detect_fvg(df, lookback=60):
    """
    ناحیه‌های Fair Value Gap پرنشده (unfilled) در N کندل اخیر را پیدا می‌کند.
    خروجی: (bullish_gaps, bearish_gaps) که هرکدام لیستی از دیکشنری
    {'low':..., 'high':..., 'index':...} هستند (نزدیک‌ترین به انتها آخر لیست است).
    """
    n = len(df)
    start = max(1, n - lookback)
    bullish_gaps, bearish_gaps = [], []

    for i in range(start, n - 1):
        prev_high = df["high"].iloc[i - 1]
        prev_low = df["low"].iloc[i - 1]
        next_high = df["high"].iloc[i + 1]
        next_low = df["low"].iloc[i + 1]
        after = df.iloc[i + 2 :] if i + 2 < n else df.iloc[0:0]

        if prev_high < next_low:  # شکاف صعودی (Bullish FVG)
            gap_low, gap_high = prev_high, next_low
            filled = (not after.empty) and (after["low"] <= gap_high).any()
            if not filled:
                bullish_gaps.append({"low": gap_low, "high": gap_high, "index": i})

        elif prev_low > next_high:  # شکاف نزولی (Bearish FVG)
            gap_low, gap_high = next_high, prev_low
            filled = (not after.empty) and (after["high"] >= gap_low).any()
            if not filled:
                bearish_gaps.append({"low": gap_low, "high": gap_high, "index": i})

    return bullish_gaps, bearish_gaps


def nearest_fvg_reaction(bullish_gaps, bearish_gaps, current_price, atr, proximity_mult=1.2):
    """
    بررسی می‌کند آیا قیمت فعلی داخل یا در نزدیکی یکی از FVGهای پرنشده است؛
    این وضعیت معمولا به‌عنوان واکنش احتمالی (بازگشت/حمایت/مقاومت) در نظر گرفته می‌شود.
    خروجی: 'BULLISH_FVG_REACTION', 'BEARISH_FVG_REACTION' یا None
    """
    proximity = atr * proximity_mult if atr else 0

    for gap in reversed(bullish_gaps):
        if gap["low"] - proximity <= current_price <= gap["high"] + proximity:
            return "BULLISH_FVG_REACTION"

    for gap in reversed(bearish_gaps):
        if gap["low"] - proximity <= current_price <= gap["high"] + proximity:
            return "BEARISH_FVG_REACTION"

    return None


def detect_order_blocks(df, swing_highs, swing_lows, lookback=80, displacement_mult=1.5):
    """
    آخرین Order Block معتبر صعودی و نزولی را در N کندل اخیر پیدا می‌کند.
    خروجی: (bullish_ob, bearish_ob) که هرکدام None یا دیکشنری
    {'low':..., 'high':..., 'index':...} هستند.
    """
    n = len(df)
    start = max(15, n - lookback)
    bullish_ob, bearish_ob = None, None

    for i in range(start, n):
        candle = df.iloc[i]
        prev_candle = df.iloc[i - 1]
        window_start = max(0, i - 14)
        avg_range = (df["high"].iloc[window_start:i] - df["low"].iloc[window_start:i]).mean()
        if not avg_range or avg_range <= 0:
            continue

        candle_move = candle["close"] - candle["open"]
        is_prev_bearish = prev_candle["close"] < prev_candle["open"]
        is_prev_bullish = prev_candle["close"] > prev_candle["open"]

        # Order Block صعودی: کندل نزولی قبل از یک کندل جابه‌جایی صعودی قوی که سقف اخیر را می‌شکند
        if is_prev_bearish and candle_move > displacement_mult * avg_range:
            recent_highs = [p for idx, p in swing_highs if idx < i]
            if recent_highs and candle["close"] > recent_highs[-1]:
                bullish_ob = {"low": prev_candle["low"], "high": prev_candle["high"], "index": i - 1}

        # Order Block نزولی: کندل صعودی قبل از یک کندل جابه‌جایی نزولی قوی که کف اخیر را می‌شکند
        if is_prev_bullish and -candle_move > displacement_mult * avg_range:
            recent_lows = [p for idx, p in swing_lows if idx < i]
            if recent_lows and candle["close"] < recent_lows[-1]:
                bearish_ob = {"low": prev_candle["low"], "high": prev_candle["high"], "index": i - 1}

    return bullish_ob, bearish_ob


def order_block_reaction(bullish_ob, bearish_ob, current_price, atr, proximity_mult=0.5):
    """بررسی می‌کند آیا قیمت به یکی از Order Blockهای شناسایی‌شده بازگشته است."""
    proximity = atr * proximity_mult if atr else 0

    if bullish_ob and (bullish_ob["low"] - proximity) <= current_price <= (bullish_ob["high"] + proximity):
        return "BULLISH_OB_REACTION"
    if bearish_ob and (bearish_ob["low"] - proximity) <= current_price <= (bearish_ob["high"] + proximity):
        return "BEARISH_OB_REACTION"
    return None

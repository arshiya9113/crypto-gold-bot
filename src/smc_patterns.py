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


def detect_liquidity_sweep(df, swing_highs, swing_lows):
    """
    شکار نقدینگی / استاپ‌هانت (Liquidity Sweep): قیمت با سایه از یک سقف یا کف
    سوینگ اخیر عبور می‌کند (استاپ‌لاس‌های خوابیده آنجا را می‌گیرد) ولی با بدنه
    کندل به داخل محدوده قبلی بازمی‌گردد. این یکی از قوی‌ترین سیگنال‌های بازگشتی
    در مکتب پول هوشمند است، چون نشان می‌دهد نقدینگی جمع‌آوری شده و قیمت رد شده.

    خروجی: 'BEARISH_LIQUIDITY_SWEEP' (سوئیپ سقف - هشدار نزولی)،
            'BULLISH_LIQUIDITY_SWEEP' (سوئیپ کف - هشدار صعودی) یا None
    """
    if len(df) < 2:
        return None
    last = df.iloc[-1]

    for _, level in swing_highs[-3:]:
        if last["high"] > level and last["close"] < level:
            return "BEARISH_LIQUIDITY_SWEEP"

    for _, level in swing_lows[-3:]:
        if last["low"] < level and last["close"] > level:
            return "BULLISH_LIQUIDITY_SWEEP"

    return None


def detect_equal_highs_lows(swing_highs, swing_lows, tolerance=0.0015):
    """
    سقف‌ها یا کف‌های برابر (Equal Highs / Equal Lows): چند سوینگ نزدیک به هم در
    یک سطح، نشانه تجمع نقدینگی (استاپ‌های بازار) که اغلب هدف بعدی حرکت قیمت است
    (نه لزوما جهت نهایی). خروجی: {'equal_highs': level یا None, 'equal_lows': level یا None}
    """
    result = {"equal_highs": None, "equal_lows": None}

    if len(swing_highs) >= 2:
        h1, h2 = swing_highs[-2][1], swing_highs[-1][1]
        if abs(h1 - h2) / max(h1, h2) < tolerance:
            result["equal_highs"] = max(h1, h2)

    if len(swing_lows) >= 2:
        l1, l2 = swing_lows[-2][1], swing_lows[-1][1]
        if abs(l1 - l2) / max(l1, l2) < tolerance:
            result["equal_lows"] = min(l1, l2)

    return result


def compute_premium_discount_zone(swing_highs, swing_lows):
    """
    بر اساس آخرین لگ قیمتی (بین آخرین سقف و کف سوینگ)، سطح تعادل (Equilibrium)
    را محاسبه می‌کند. در مکتب پول هوشمند، نیمه بالایی لگ «پرمیوم» (Premium - منطقه
    ترجیحی فروش) و نیمه پایینی «دیسکانت» (Discount - منطقه ترجیحی خرید) نام دارد.
    """
    if not swing_highs or not swing_lows:
        return None
    last_high = swing_highs[-1][1]
    last_low = swing_lows[-1][1]
    if last_high <= last_low:
        return None
    return {"high": last_high, "low": last_low, "equilibrium": (last_high + last_low) / 2}


def price_zone(zone, current_price):
    """موقعیت قیمت را نسبت به ناحیه پرمیوم/دیسکانت مشخص می‌کند."""
    if not zone:
        return None
    return "PREMIUM" if current_price >= zone["equilibrium"] else "DISCOUNT"


def detect_breaker_block(df, bullish_ob, bearish_ob):
    """
    بریکر بلاک (Breaker Block): وقتی یک Order Block به‌طور قاطع شکسته می‌شود
    (بسته شدن قیمت آن‌طرف ناحیه)، آن نقش خودش را معکوس می‌کند - یک OB صعودی
    شکسته‌شده به ناحیه مقاومت تبدیل می‌شود (و بالعکس برای OB نزولی).
    """
    if len(df) == 0:
        return None
    close = df["close"].iloc[-1]

    if bullish_ob and close < bullish_ob["low"]:
        return {"type": "BEARISH_BREAKER", "low": bullish_ob["low"], "high": bullish_ob["high"]}
    if bearish_ob and close > bearish_ob["high"]:
        return {"type": "BULLISH_BREAKER", "low": bearish_ob["low"], "high": bearish_ob["high"]}
    return None


def breaker_block_reaction(breaker, current_price, atr, proximity_mult=0.5):
    """بررسی می‌کند آیا قیمت به ناحیه بریکر بلاک بازگشته است."""
    if not breaker:
        return None
    proximity = atr * proximity_mult if atr else 0
    if (breaker["low"] - proximity) <= current_price <= (breaker["high"] + proximity):
        return breaker["type"] + "_REACTION"
    return None


def resample_ohlcv(df, rule="4h"):
    """کندل‌های تایم‌فریم پایین را به یک تایم‌فریم بالاتر تبدیل می‌کند (بدون نیاز
    به درخواست شبکه جدید) تا امکان تحلیل چند تایم‌فریمی (Top-Down) فراهم شود."""
    working = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    working = working.set_index("timestamp")
    resampled = working.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return resampled.reset_index()


def compute_htf_bias(df, rule="4h", fast=20, slow=50):
    """
    با بازسازی داده تایم‌فریم پایین به یک تایم‌فریم بالاتر (مثلا ۴ ساعته از روی
    داده ۱ ساعته)، جهت‌گیری کلی‌تر بازار را برای تایید/رد سیگنال تایم‌فریم پایین
    محاسبه می‌کند (اصل تحلیل بالا-به-پایین / Top-Down در مکتب پول هوشمند).
    خروجی: 'BULLISH'، 'BEARISH' یا None (داده کافی نبود)
    """
    try:
        htf = resample_ohlcv(df, rule=rule)
    except Exception:  # noqa: BLE001
        return None

    if len(htf) < slow + 5:
        return None

    ema_fast = htf["close"].ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = htf["close"].ewm(span=slow, adjust=False).mean().iloc[-1]

    if ema_fast > ema_slow:
        return "BULLISH"
    if ema_fast < ema_slow:
        return "BEARISH"
    return None


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

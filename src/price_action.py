"""
تحلیل پرایس اکشن (Price Action):
- تشخیص نقاط سوینگ (Swing High/Low) به روش فرکتال
- تشخیص ساختار روند (Higher Highs/Higher Lows یا Lower Highs/Lower Lows)
- تشخیص الگوهای کندل استیک رایج (انگالفینگ، همر، شوتینگ استار، دوجی، مورنینگ/ایونینگ استار)
- تشخیص شکست کانال دونچیان (Donchian Breakout)
- یافتن نزدیک‌ترین سطوح حمایت/مقاومت نسبت به قیمت فعلی
"""


def detect_volatility_squeeze(df, lookback=6):
    """
    فشردگی نوسان (Volatility Squeeze / TTM Squeeze): وقتی باند بولینگر کاملا داخل
    کانال کلتنر قرار می‌گیرد یعنی نوسان به‌شدت افت کرده - این وضعیت معمولا قبل از
    یک حرکت انفجاری (به هر دو سمت) اتفاق می‌افتد. جهت حرکت از این تابع مشخص نیست،
    فقط هشدار می‌دهد که "چیزی در حال شکل‌گیری است".

    خروجی:
      'SQUEEZE_ON'       -> نوسان همین الان فشرده است (در حال آماده‌سازی)
      'SQUEEZE_RELEASED' -> همین چند کندل پیش فشرده بود و تازه آزاد شده (لحظه احتمالی شروع حرکت)
      None               -> وضعیت خاصی نیست
    """
    required = {"bb_high", "bb_low", "kc_high", "kc_low"}
    if not required.issubset(df.columns) or len(df) < lookback + 1:
        return None

    squeeze_series = (df["bb_high"] < df["kc_high"]) & (df["bb_low"] > df["kc_low"])
    squeeze_series = squeeze_series.dropna()
    if len(squeeze_series) < lookback + 1:
        return None

    currently_on = bool(squeeze_series.iloc[-1])
    was_on_recently = bool(squeeze_series.iloc[-lookback - 1 : -1].any())

    if currently_on:
        return "SQUEEZE_ON"
    if was_on_recently:
        return "SQUEEZE_RELEASED"
    return None


def detect_accumulation_signal(df, lookback=5, volume_mult=1.8, max_range_pct=0.025):
    """
    انباشت/توزیع مشکوک: حجم معاملات به‌طور محسوسی بالاتر از میانگین است ولی قیمت
    در یک بازه بسیار کوچک نوسان کرده - نشانه‌ای از ورود سرمایه سنگین بدون حرکت
    قابل توجه قیمت، که می‌تواند مقدمه یک حرکت شارپ باشد.
    """
    if "vol_sma20" not in df.columns or len(df) < lookback + 1:
        return False

    recent = df.iloc[-lookback:]
    if recent["vol_sma20"].isna().any() or (recent["vol_sma20"] <= 0).any():
        return False

    avg_volume_ratio = (recent["volume"] / recent["vol_sma20"]).mean()
    last_close = recent["close"].iloc[-1]
    if last_close == 0:
        return False
    price_range_pct = (recent["high"].max() - recent["low"].min()) / last_close

    return bool(avg_volume_ratio > volume_mult and price_range_pct < max_range_pct)


def find_swing_points(df, order=3):
    """
    نقاط سوینگ های/لو را با روش فرکتال (مقایسه با N کندل قبل و بعد) پیدا می‌کند.
    خروجی: (swing_highs, swing_lows) که هرکدام لیستی از (index, price) هستند.
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_highs, swing_lows = [], []

    for i in range(order, n - order):
        window_high = highs[i - order : i + order + 1]
        if highs[i] == window_high.max() and list(window_high).index(window_high.max()) == order:
            swing_highs.append((i, highs[i]))

        window_low = lows[i - order : i + order + 1]
        if lows[i] == window_low.min() and list(window_low).index(window_low.min()) == order:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def detect_trend_structure(swing_highs, swing_lows):
    """بر اساس آخرین سوینگ‌ها، ساختار روند را تشخیص می‌دهد."""
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "UNKNOWN", 0

    last_highs = [p for _, p in swing_highs[-3:]]
    last_lows = [p for _, p in swing_lows[-3:]]

    higher_highs = all(last_highs[i] < last_highs[i + 1] for i in range(len(last_highs) - 1))
    higher_lows = all(last_lows[i] < last_lows[i + 1] for i in range(len(last_lows) - 1))
    lower_highs = all(last_highs[i] > last_highs[i + 1] for i in range(len(last_highs) - 1))
    lower_lows = all(last_lows[i] > last_lows[i + 1] for i in range(len(last_lows) - 1))

    if higher_highs and higher_lows:
        return "UPTREND", 15
    if lower_highs and lower_lows:
        return "DOWNTREND", -15
    return "RANGE", 0


def detect_candlestick_pattern(df):
    """الگوهای دو-کندلی و تک‌کندلی رایج را روی آخرین کندل تشخیص می‌دهد."""
    if len(df) < 2:
        return []

    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"] if last["high"] != last["low"] else 1e-9
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]

    patterns = []

    if prev["close"] < prev["open"] and last["close"] > last["open"] \
            and last["close"] >= prev["open"] and last["open"] <= prev["close"]:
        patterns.append(("انگالفینگ صعودی (Bullish Engulfing)", 10))

    if prev["close"] > prev["open"] and last["close"] < last["open"] \
            and last["open"] >= prev["close"] and last["close"] <= prev["open"]:
        patterns.append(("انگالفینگ نزولی (Bearish Engulfing)", -10))

    if body / candle_range < 0.35 and lower_wick > body * 2 and upper_wick < body * 0.8:
        patterns.append(("همر (Hammer)", 7))

    if body / candle_range < 0.35 and upper_wick > body * 2 and lower_wick < body * 0.8:
        patterns.append(("شوتینگ استار (Shooting Star)", -7))

    if body / candle_range < 0.1:
        patterns.append(("دوجی (Doji - بلاتکلیفی بازار)", 0))

    return patterns


def detect_star_patterns(df):
    """الگوهای سه‌کندلی مورنینگ استار و ایونینگ استار را تشخیص می‌دهد."""
    if len(df) < 3:
        return []

    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    patterns = []

    body_c1 = abs(c1["close"] - c1["open"])
    body_c2 = abs(c2["close"] - c2["open"])

    c1_bearish = c1["close"] < c1["open"]
    c1_bullish = c1["close"] > c1["open"]
    c3_bullish = c3["close"] > c3["open"]
    c3_bearish = c3["close"] < c3["open"]
    small_middle_body = body_c2 < body_c1 * 0.5

    if c1_bearish and small_middle_body and c3_bullish and c3["close"] > (c1["open"] + c1["close"]) / 2:
        patterns.append(("مورنینگ استار (Morning Star)", 12))

    if c1_bullish and small_middle_body and c3_bearish and c3["close"] < (c1["open"] + c1["close"]) / 2:
        patterns.append(("ایونینگ استار (Evening Star)", -12))

    return patterns


def donchian_breakout(df, period=20):
    """بررسی می‌کند آیا کندل آخر از سقف/کف N کندل قبلی خودش عبور کرده یا نه."""
    if len(df) < period + 1:
        return "NONE", 0

    highest = df["high"].iloc[-period - 1 : -1].max()
    lowest = df["low"].iloc[-period - 1 : -1].min()
    close = df["close"].iloc[-1]

    if close > highest:
        return "UP", 15
    if close < lowest:
        return "DOWN", -15
    return "NONE", 0


def nearest_levels(swing_highs, swing_lows, current_price):
    """نزدیک‌ترین سطح حمایت (زیر قیمت) و مقاومت (بالای قیمت) را برمی‌گرداند."""
    resistances = sorted(p for _, p in swing_highs if p > current_price)
    supports = sorted((p for _, p in swing_lows if p < current_price), reverse=True)

    nearest_resistance = resistances[0] if resistances else None
    nearest_support = supports[0] if supports else None
    return nearest_support, nearest_resistance


def fibonacci_levels(swing_high, swing_low):
    """سطوح فیبوناچی ریتریسمنت بین یک سقف و کف مشخص را محاسبه می‌کند."""
    diff = swing_high - swing_low
    return {
        "0.0": swing_high,
        "0.236": swing_high - 0.236 * diff,
        "0.382": swing_high - 0.382 * diff,
        "0.5": swing_high - 0.5 * diff,
        "0.618": swing_high - 0.618 * diff,
        "0.786": swing_high - 0.786 * diff,
        "1.0": swing_low,
    }

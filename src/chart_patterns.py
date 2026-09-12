"""
الگوهای کلاسیک تحلیل تکنیکال چارتی، همه بر پایه نقاط سوینگ (خروجی
price_action.find_swing_points) و کندل‌های خام محاسبه می‌شوند.
"""
import numpy as np


def detect_head_and_shoulders(swing_highs, swing_lows, tolerance=0.035):
    """
    الگوی سر و شانه (نزولی، Bearish Reversal): سه سقف متوالی که سقف وسط
    (سر) به‌وضوح از دو شانه بالاتر باشد و دو شانه تقریبا هم‌ارتفاع باشند.
    خروجی: دیکشنری {'neckline':..., 'head_index':..., 'right_shoulder_index':...} یا None
    """
    if len(swing_highs) < 3 or len(swing_lows) < 2:
        return None

    (i1, left), (i2, head), (i3, right) = swing_highs[-3:]
    if not (head > left and head > right):
        return None
    if abs(left - right) / head > tolerance:
        return None
    if (head - max(left, right)) / head < 0.01:  # سر باید محسوس بالاتر باشد
        return None

    neckline_points = [p for idx, p in swing_lows if i1 < idx < i3]
    if len(neckline_points) < 2:
        return None
    neckline = sum(neckline_points[:2]) / 2

    return {"neckline": neckline, "head_index": i2, "right_shoulder_index": i3}


def detect_inverse_head_and_shoulders(swing_highs, swing_lows, tolerance=0.035):
    """نسخه معکوس (صعودی، Bullish Reversal) الگوی سر و شانه، بر پایه کف‌ها."""
    if len(swing_lows) < 3 or len(swing_highs) < 2:
        return None

    (i1, left), (i2, head), (i3, right) = swing_lows[-3:]
    if not (head < left and head < right):
        return None
    if abs(left - right) / head > tolerance:
        return None
    if (min(left, right) - head) / head < 0.01:
        return None

    neckline_points = [p for idx, p in swing_highs if i1 < idx < i3]
    if len(neckline_points) < 2:
        return None
    neckline = sum(neckline_points[:2]) / 2

    return {"neckline": neckline, "head_index": i2, "right_shoulder_index": i3}


def detect_double_top(swing_highs, swing_lows, tolerance=0.02):
    """سقف دوقلو (نزولی): دو سقف هم‌ارتفاع با یک دره بین آن‌ها."""
    if len(swing_highs) < 2:
        return None
    (i1, h1), (i2, h2) = swing_highs[-2:]
    if abs(h1 - h2) / max(h1, h2) > tolerance:
        return None
    troughs = [p for idx, p in swing_lows if i1 < idx < i2]
    if not troughs:
        return None
    return {"neckline": min(troughs), "second_top_index": i2}


def detect_double_bottom(swing_highs, swing_lows, tolerance=0.02):
    """کف دوقلو (صعودی): دو کف هم‌ارتفاع با یک قله بین آن‌ها."""
    if len(swing_lows) < 2:
        return None
    (i1, l1), (i2, l2) = swing_lows[-2:]
    if abs(l1 - l2) / max(l1, l2) > tolerance:
        return None
    peaks = [p for idx, p in swing_highs if i1 < idx < i2]
    if not peaks:
        return None
    return {"neckline": max(peaks), "second_bottom_index": i2}


def detect_triangle(swing_highs, swing_lows, flat_threshold=0.0008):
    """
    مثلث صعودی/نزولی/متقارن را با برازش خط روند بر آخرین سوینگ‌ها تشخیص می‌دهد.
    خروجی: 'ASCENDING' (صعودی)، 'DESCENDING' (نزولی)، 'SYMMETRICAL' (متقارن) یا None
    """
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return None

    hi_idx = np.array([idx for idx, _ in swing_highs[-3:]], dtype=float)
    hi_val = np.array([p for _, p in swing_highs[-3:]], dtype=float)
    lo_idx = np.array([idx for idx, _ in swing_lows[-3:]], dtype=float)
    lo_val = np.array([p for _, p in swing_lows[-3:]], dtype=float)

    avg_price = (hi_val.mean() + lo_val.mean()) / 2
    if avg_price == 0:
        return None

    slope_hi = np.polyfit(hi_idx, hi_val, 1)[0] / avg_price
    slope_lo = np.polyfit(lo_idx, lo_val, 1)[0] / avg_price

    hi_flat = abs(slope_hi) < flat_threshold
    lo_flat = abs(slope_lo) < flat_threshold

    if hi_flat and slope_lo > flat_threshold:
        return "ASCENDING"
    if slope_hi < -flat_threshold and lo_flat:
        return "DESCENDING"
    if slope_hi < -flat_threshold and slope_lo > flat_threshold:
        return "SYMMETRICAL"
    return None


def detect_channel(swing_highs, swing_lows, min_points=3, parallel_ratio_tolerance=0.5):
    """
    کانال قیمتی (صعودی/نزولی/افقی) را با برازش خط روند بر سوینگ‌های بالا و پایین
    تشخیص می‌دهد.
    خروجی: {'type':..., 'upper':..., 'lower':...} یا None
    """
    if len(swing_highs) < min_points or len(swing_lows) < min_points:
        return None

    hi_idx = np.array([idx for idx, _ in swing_highs[-min_points:]], dtype=float)
    hi_val = np.array([p for _, p in swing_highs[-min_points:]], dtype=float)
    lo_idx = np.array([idx for idx, _ in swing_lows[-min_points:]], dtype=float)
    lo_val = np.array([p for _, p in swing_lows[-min_points:]], dtype=float)

    avg_price = (hi_val.mean() + lo_val.mean()) / 2
    if avg_price == 0:
        return None

    hi_slope, hi_intercept = np.polyfit(hi_idx, hi_val, 1)
    lo_slope, lo_intercept = np.polyfit(lo_idx, lo_val, 1)

    rel_hi_slope = hi_slope / avg_price
    rel_lo_slope = lo_slope / avg_price

    # شیب‌ها باید هم‌جهت و تقریبا موازی باشند تا کانال معتبر باشد
    if rel_hi_slope * rel_lo_slope < 0:
        return None
    max_slope = max(abs(rel_hi_slope), abs(rel_lo_slope), 1e-9)
    if abs(abs(rel_hi_slope) - abs(rel_lo_slope)) / max_slope > parallel_ratio_tolerance:
        return None

    last_index = max(hi_idx.max(), lo_idx.max())
    upper_at_last = hi_slope * last_index + hi_intercept
    lower_at_last = lo_slope * last_index + lo_intercept
    if upper_at_last <= lower_at_last:
        return None

    if rel_hi_slope > 0.0008:
        channel_type = "ASCENDING"
    elif rel_hi_slope < -0.0008:
        channel_type = "DESCENDING"
    else:
        channel_type = "HORIZONTAL"

    return {"type": channel_type, "upper": upper_at_last, "lower": lower_at_last}


def channel_position(channel, current_price):
    """موقعیت نسبی قیمت فعلی داخل کانال را بین ۰ (کف) و ۱ (سقف) برمی‌گرداند."""
    if not channel:
        return None
    span = channel["upper"] - channel["lower"]
    if span <= 0:
        return None
    return max(0.0, min(1.0, (current_price - channel["lower"]) / span))


def detect_climax_spike(df, atr_col="atr14", range_mult=2.5):
    """
    کندل اسپایک/کلایمکس: کندلی با رنج بسیار بزرگ‌تر از حد معمول (نسبت به ATR).
    اگر سایه بلندی در جهت مخالف بدنه داشته باشد، نشانه رد قیمت (Rejection) و
    احتمال برگشت است؛ در غیر این صورت صرفا یک کندل مومنتومی قوی است.
    """
    if atr_col not in df.columns or len(df) < 2:
        return None
    last = df.iloc[-1]
    atr = last[atr_col]
    if not atr or np.isnan(atr) or atr <= 0:
        return None

    candle_range = last["high"] - last["low"]
    if candle_range < range_mult * atr:
        return None

    body = abs(last["close"] - last["open"])
    upper_wick = last["high"] - max(last["close"], last["open"])
    lower_wick = min(last["close"], last["open"]) - last["low"]
    is_bullish_candle = last["close"] > last["open"]

    if is_bullish_candle and upper_wick > body:
        return "BUY_CLIMAX"  # اسپایک صعودی با رد شدن از بالا -> هشدار برگشت نزولی
    if not is_bullish_candle and lower_wick > body:
        return "SELL_CLIMAX"  # اسپایک نزولی با رد شدن از پایین -> هشدار برگشت صعودی
    return "MOMENTUM_SPIKE_UP" if is_bullish_candle else "MOMENTUM_SPIKE_DOWN"


def estimate_elliott_wave_hint(swing_highs, swing_lows):
    """
    تخمین بسیار ساده و غیرقطعی، صرفا جهت آگاهی اضافه - نه شمارش دقیق الیوت.
    شمارش واقعی امواج الیوت نیازمند بررسی نسبت‌های فیبوناچی، قواعد اعتبارسنجی موج
    (مثلا موج ۳ نباید کوتاه‌ترین باشد) و اغلب تفسیر چندگانه توسط تحلیلگران مختلف
    است؛ به همین دلیل این تابع فقط یک الگوی ساده تناوب ۵ سوینگ آخر را بررسی
    می‌کند و در امتیازدهی نهایی وزن بسیار کمی به آن داده می‌شود.
    """
    combined = sorted(swing_highs + swing_lows, key=lambda x: x[0])
    if len(combined) < 5:
        return None

    last5 = combined[-5:]
    diffs = [last5[i + 1][1] - last5[i][1] for i in range(4)]
    directions = [1 if d > 0 else -1 for d in diffs]
    is_alternating = all(directions[i] != directions[i + 1] for i in range(len(directions) - 1))
    if not is_alternating:
        return None

    return "IMPULSE_UP" if directions[0] == 1 else "IMPULSE_DOWN" 
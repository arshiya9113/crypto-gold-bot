"""
شاخص فشار هوشمند (Smart Pressure Index - SPI)
================================================
این یک اندیکاتور اختصاصی و طراحی‌شده برای این ربات است، نه یکی از اندیکاتورهای
استاندارد و شناخته‌شده بازار (مثل RSI یا MACD). هدف آن دیدن چیزی است که
اندیکاتورهای رایج به‌تنهایی نمی‌بینند: ترکیب هم‌زمان "محل بسته شدن کندل داخل
رنج آن" (قدرت تصمیم بازار)، "عدم‌تقارن سایه‌ها" (رد شدن قیمت از یک جهت - دقیقا
جایی که سفارش‌های خرد توسط بازیگران بزرگ جذب می‌شود) و "وزن نسبی حجم" (اهمیت
واقعی هر کندل نسبت به میانگین اخیر).

فرمول هر کندل:
  body_position  = (close - open) / range          -> بین تقریبا -1 تا +1
  wick_asymmetry = (سایه پایین - سایه بالا) / range -> عدم‌تقارن رد شدن قیمت
  raw_pressure   = 0.6 * body_position + 0.4 * wick_asymmetry
  weighted       = raw_pressure * min(volume / میانگین حجم ۲۰ کندل, سقف ۴)

از "weighted" یک نسخه هموارشده (EMA) به‌عنوان نوسان‌گر SPI و یک نسخه تجمعی
(Cumulative Sum) به‌عنوان خط انباشت SPI ساخته می‌شود. مهم‌ترین کاربرد این
اندیکاتور، تشخیص "دیورجنس" بین خط انباشت SPI و قیمت است: وقتی قیمت سقف بالاتر
می‌زند ولی خط انباشت SPI سقف پایین‌تر می‌زند (یا برعکس در کف‌ها)، نشانه ضعف/قدرت
پنهانی است که در خود نمودار قیمت هنوز دیده نمی‌شود.

⚠️ این یک اندیکاتور تجربی و اختصاصی است، نه یک روش دارای اعتبار آکادمیک یا
سابقه چند دهه‌ای مثل RSI/MACD. مثل بقیه بخش‌های این ربات، عملکرد واقعی‌اش باید
با بک‌تست سنجیده شود.
"""
import numpy as np


def compute_spi(df, volume_window=20, smooth=5, volume_cap=4.0, accumulation_window=50):
    """
    شاخص فشار هوشمند خام، هموارشده و انباشتی (پنجره‌ای، نه از ابتدای بی‌نهایت
    داده) را محاسبه می‌کند. خروجی: (raw, smoothed, accumulated)
    از یک پنجره غلتان (نه cumsum نامحدود از ابتدای دیتافریم) برای خط انباشتی
    استفاده می‌شود تا مقایسه بین دو سوینگ به طول اختیاری تاریخچه داده حساس نباشد.
    """
    high_low_range = (df["high"] - df["low"])
    epsilon = (df["close"].abs() * 1e-6).clip(lower=1e-9)
    safe_range = high_low_range.where(high_low_range > 0, epsilon)

    body_position = (df["close"] - df["open"]) / safe_range

    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    wick_asymmetry = (lower_wick - upper_wick) / safe_range

    raw_pressure = 0.6 * body_position + 0.4 * wick_asymmetry

    volume_avg = df["volume"].rolling(volume_window).mean()
    volume_ratio = (df["volume"] / volume_avg).clip(upper=volume_cap)
    volume_ratio = volume_ratio.fillna(1.0)

    weighted = raw_pressure * volume_ratio
    smoothed = weighted.ewm(span=smooth, adjust=False).mean()
    accumulated = weighted.rolling(window=accumulation_window, min_periods=5).sum()

    return weighted, smoothed, accumulated


def detect_spi_divergence(swing_highs, swing_lows, spi_accumulated):
    """
    دیورجنس بین خط انباشتی پنجره‌ای SPI و قیمت را با استفاده از همان نقاط
    سوینگ قیمتی (که برای بقیه تحلیل پرایس‌اکشن هم استفاده می‌شود) بررسی می‌کند.

    خروجی: 'BEARISH_DIVERGENCE'، 'BULLISH_DIVERGENCE' یا None
    """
    if len(swing_highs) >= 2:
        (i1, p1), (i2, p2) = swing_highs[-2:]
        if p2 > p1 and i2 < len(spi_accumulated) and i1 < len(spi_accumulated):
            if spi_accumulated.iloc[i2] < spi_accumulated.iloc[i1]:
                return "BEARISH_DIVERGENCE"

    if len(swing_lows) >= 2:
        (i1, p1), (i2, p2) = swing_lows[-2:]
        if p2 < p1 and i2 < len(spi_accumulated) and i1 < len(spi_accumulated):
            if spi_accumulated.iloc[i2] > spi_accumulated.iloc[i1]:
                return "BULLISH_DIVERGENCE"

    return None


def detect_spi_exhaustion(spi_smoothed, window=50, z_threshold=1.8):
    """
    بررسی می‌کند آیا فشار خرید/فروش فعلی (بر مبنای SPI هموارشده) نسبت به
    رفتار اخیر خودش به‌طور غیرعادی زیاد شده - نشانه احتمالی خستگی حرکت و
    اصلاح نزدیک (مفهومی شبیه اشباع خرید/فروش در RSI، ولی حجم و سایه هم لحاظ شده).
    """
    if len(spi_smoothed) < window:
        return None

    recent = spi_smoothed.iloc[-window:]
    mean = recent.mean()
    std = recent.std()
    if not std or np.isnan(std):
        return None

    z_score = (spi_smoothed.iloc[-1] - mean) / std
    if z_score > z_threshold:
        return "OVERHEATED_BUY_PRESSURE"
    if z_score < -z_threshold:
        return "OVERHEATED_SELL_PRESSURE"
    return None

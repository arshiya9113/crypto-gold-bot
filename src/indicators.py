"""
محاسبه مجموعه‌ای کامل از اندیکاتورهای تحلیل تکنیکال با استفاده از کتابخانه ta:
روند (EMA, MACD, ADX)، مومنتوم (RSI, StochRSI, CCI)، نوسان (Bollinger, ATR) و حجم (OBV).
"""
import numpy as np
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def compute_indicators(df):
    df = df.copy()

    df["ema9"] = EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema21"] = EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    adx_ind = ADXIndicator(df["high"], df["low"], df["close"])
    df["adx"] = adx_ind.adx()
    df["plus_di"] = adx_ind.adx_pos()
    df["minus_di"] = adx_ind.adx_neg()

    df["rsi14"] = RSIIndicator(df["close"], window=14).rsi()

    try:
        stoch_rsi = StochRSIIndicator(df["close"])
        df["stoch_rsi_k"] = stoch_rsi.stochrsi_k()
        df["stoch_rsi_d"] = stoch_rsi.stochrsi_d()
    except Exception:  # noqa: BLE001
        df["stoch_rsi_k"] = np.nan
        df["stoch_rsi_d"] = np.nan

    df["cci"] = CCIIndicator(df["high"], df["low"], df["close"]).cci()

    bb = BollingerBands(df["close"])
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()

    df["atr14"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

    df["obv"] = OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["vol_sma20"] = df["volume"].rolling(20).mean()

    return df

 """
Binance Alpha & Spot technical levels (V3).
Pass candles_are_closed=True when input contains only completed candles.
"""
import math
import pandas as pd


def technical_levels(df, lookback=60, atr_period=14, candles_are_closed=False):
    """Return indicative long-side levels, or None for insufficient/invalid data."""
    if df is None or not {"high", "low", "close"}.issubset(df.columns):
        return None
    data = df.copy()
    for col in ("high", "low", "close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["high", "low", "close"])
    if not candles_are_closed:
        data = data.iloc[:-1]
    if len(data) < max(lookback + 1, atr_period + 2):
        return None
    if ((data[["high", "low", "close"]] <= 0).any().any()
            or (data["high"] < data["low"]).any()
            or (data["close"] > data["high"]).any()
            or (data["close"] < data["low"]).any()):
        return None

    high, low, close = data["high"], data["low"], data["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = float(true_range.ewm(
        alpha=1 / atr_period, adjust=False, min_periods=atr_period
    ).mean().iloc[-1])
    price = float(close.iloc[-1])
    history = data.iloc[-(lookback + 1):-1]
    support = float(history["low"].min())
    resistance = float(history["high"].max())
    if not all(math.isfinite(x) and x > 0 for x in (price, atr, support, resistance)):
        return None
    if support >= resistance:
        return None

    result = dict(price=price, support=support, resistance=resistance,
                  atr=atr, signal="OUTSIDE_RANGE", entry=None, stop=None,
                  target1=None, target2=None, risk_reward=None)
    if not support < price < resistance:
        return result

    if (price - support) / atr <= 1:
        signal = "NEAR_SUPPORT"
    elif (resistance - price) / atr <= 1:
        signal = "NEAR_RESISTANCE"
    else:
        signal = "NEUTRAL"
    entry = price
    stop = min(support - 0.5 * atr, entry - 1.5 * atr)
    risk = entry - stop
    target1 = min(resistance, entry + 2 * risk)
    target2 = min(resistance, entry + 3 * risk)
    reward = target1 - entry
    if not (0 < stop < entry < target1) or risk <= 0 or reward <= 0:
        return None
    result.update(signal=signal, entry=entry, stop=stop,
                  target1=target1, target2=target2, risk_reward=reward / risk)
    return result

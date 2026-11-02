"""Binance Alpha & Spot technical levels (V4).

Uses completed OHLC candles. Levels are indicative, not trade instructions.
The caller must independently confirm volume, trend and signal freshness.
"""
import math
import pandas as pd


def technical_levels(df, lookback=60, atr_period=14, candles_are_closed=False):
    """Return levels with the legacy keys used by existing scanners.

    Signal names: NEAR_SUPPORT, NEAR_RESISTANCE, BREAKOUT_CANDIDATE,
    BREAKDOWN, NEUTRAL, OUTSIDE_RANGE. Only actionable long-side setups
    receive entry, stop and targets; breakout requires external volume check.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    if not {"high", "low", "close"}.issubset(df.columns):
        return None
    if not isinstance(lookback, int) or not isinstance(atr_period, int):
        return None
    if lookback < 20 or atr_period < 2:
        return None

    data = df.copy()
    for col in ("high", "low", "close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # Do not silently bridge malformed candles or missing observations.
    if data[["high", "low", "close"]].isna().any().any():
        return None
    if not candles_are_closed:
        data = data.iloc[:-1]
    if len(data) < max(lookback + 2, atr_period + 2):
        return None
    if ((data[["high", "low", "close"]] <= 0).any().any()
            or (data["high"] < data["low"]).any()
            or (data["close"] > data["high"]).any()
            or (data["close"] < data["low"]).any()):
        return None

    high, low, close = data["high"], data["low"], data["close"]
    previous_close = close.shift(1)
    tr = pd.concat([high - low, (high - previous_close).abs(),
                    (low - previous_close).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / atr_period, adjust=False,
                       min_periods=atr_period).mean().iloc[-1])
    price = float(close.iloc[-1])
    previous_price = float(close.iloc[-2])
    history = data.iloc[-(lookback + 1):-1]
    support = float(history["low"].min())
    resistance = float(history["high"].max())
    if not all(math.isfinite(x) and x > 0 for x in
               (price, previous_price, atr, support, resistance)):
        return None
    if support >= resistance:
        return None

    result = dict(price=price, support=support, resistance=resistance,
                  atr=atr, signal="OUTSIDE_RANGE", entry=None, stop=None,
                  target1=None, target2=None, risk_reward=None)

    # A close above the historical range is a breakout *candidate* only.
    # Volume confirmation must be performed by the calling scanner.
    if price > resistance and previous_price <= resistance:
        signal = "BREAKOUT_CANDIDATE"
        entry = price
        stop = max(0.0, resistance - 0.5 * atr)
        risk = entry - stop
        if stop <= 0 or risk <= 0:
            return result
        target1 = entry + 2 * risk
        target2 = entry + 3 * risk
        # Do not manufacture a resistance target above the observed range.
        result.update(signal=signal, entry=entry, stop=stop,
                      target1=target1, target2=target2,
                      risk_reward=(target1 - entry) / risk)
        return result

    if price <= support:
        result["signal"] = "BREAKDOWN"
        return result
    if price >= resistance:
        return result

    if (price - support) / atr <= 1:
        signal = "NEAR_SUPPORT"
    elif (resistance - price) / atr <= 1:
        signal = "NEAR_RESISTANCE"
    else:
        signal = "NEUTRAL"
    result["signal"] = signal
    # Keep the original range-based calculations for scanner compatibility.
    entry = price
    stop = min(support - 0.5 * atr, entry - 1.5 * atr)
    risk = entry - stop
    target1 = min(resistance, entry + 2 * risk)
    target2 = min(resistance, entry + 3 * risk)
    reward = target1 - entry
    if not (0 < stop < entry < target1) or risk <= 0 or reward <= 0:
        return result
    result.update(entry=entry, stop=stop, target1=target1,
                  target2=target2, risk_reward=reward / risk)
    return result

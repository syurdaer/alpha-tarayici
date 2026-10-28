
"""
Binance Alpha & Spot - Technical Analysis V2
Support/resistance, ATR and risk/reward analysis.

Uses completed candles only.
No trading orders or Telegram messages are sent.
"""

import math
import pandas as pd


def technical_levels(df, lookback=60, atr_period=14):
    """Calculate indicative trading levels from OHLC data."""

    if df is None or len(df) < max(lookback + 1, atr_period + 2):
        return None

    data = df.copy()

    for col in ("high", "low", "close"):
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["high", "low", "close"])

    if len(data) < max(lookback + 1, atr_period + 2):
        return None

    # Exclude the latest potentially unfinished candle.
    data = data.iloc[:-1].copy()

    high = data["high"]
    low = data["low"]
    close = data["close"]

    prev_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_series = true_range.ewm(
        alpha=1 / atr_period,
        adjust=False,
        min_periods=atr_period,
    ).mean()

    atr = float(atr_series.iloc[-1])
    price = float(close.iloc[-1])

    # Historical levels exclude the current completed candle.
    history = data.iloc[-(lookback + 1):-1]

    support = float(history["low"].min())
    resistance = float(history["high"].max())

    if not all(
        math.isfinite(x) and x > 0
        for x in (price, atr, support, resistance)
    ):
        return None

    if support >= resistance:
        return None

    # Only construct a long setup when price is inside the range.
    if not support < price < resistance:
        return {
            "price": price,
            "support": support,
            "resistance": resistance,
            "atr": atr,
            "signal": "OUTSIDE_RANGE",
            "entry": None,
            "stop": None,
            "target1": None,
            "target2": None,
            "risk_reward": None,
        }

    distance_to_support = (price - support) / atr
    distance_to_resistance = (resistance - price) / atr

    # Support proximity is not a confirmed buy signal.
    if distance_to_support <= 1:
        signal = "NEAR_SUPPORT"
    elif distance_to_resistance <= 1:
        signal = "NEAR_RESISTANCE"
    else:
        signal = "NEUTRAL"

    entry = price
    stop = min(support - 0.5 * atr, entry - 1.5 * atr)

    risk = entry - stop
    target1 = min(resistance, entry + 2 * risk)
    target2 = min(resistance, entry + 3 * risk)

    reward = target1 - entry

    if risk <= 0 or reward <= 0:
        return None

    risk_reward = reward / risk

    return {
        "price": price,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "signal": signal,
        "entry": entry,
        "stop": stop,
        "target1": target1,
        "target2": target2,
        "risk_reward": risk_reward,
    }

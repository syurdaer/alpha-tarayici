"""Binance Spot scanner: confirmed candles, risk filters and deduplicated alerts."""
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import technical_analysis

BASE = "https://data-api.binance.vision/api/v3"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SpotScanner/2.0"})
TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DRY_RUN = os.getenv("DRY_RUN", "1") != "0"
MIN_VOLUME = float(os.getenv("SPOT_MIN_VOLUME", "5000000"))
MAX_COINS = int(os.getenv("SPOT_MAX_COINS", "100"))
MIN_SCORE = int(os.getenv("SPOT_MIN_SCORE", "6"))
MIN_RR = float(os.getenv("SPOT_MIN_RR", "1.5"))
STATE_FILE = Path(os.getenv("SPOT_ALERT_STATE_FILE", "spot_alert_state.json"))
INTERVAL_MS = {"1h": 3600000, "4h": 14400000}


def api(path, params=None):
    response = SESSION.get(f"{BASE}/{path}", params=params, timeout=25)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and "code" in data:
        raise ValueError(f"Binance API: {data.get('msg', data['code'])}")
    return data


def get_symbols():
    exchange = api("exchangeInfo")
    allowed = {
        item["symbol"] for item in exchange["symbols"]
        if item.get("status") == "TRADING"
        and item.get("quoteAsset") == "USDT"
        and item.get("isSpotTradingAllowed", True)
    }
    tickers = api("ticker/24hr")
    ranked = []
    for ticker in tickers:
        symbol = ticker.get("symbol", "")
        if symbol not in allowed:
            continue
        try:
            volume = float(ticker["quoteVolume"])
        except (KeyError, ValueError, TypeError):
            continue
        if math.isfinite(volume) and volume >= MIN_VOLUME:
            ranked.append((symbol, volume))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in ranked[:MAX_COINS]]


def get_klines(symbol, interval):
    rows = api("klines", {"symbol": symbol, "interval": interval, "limit": 150})
    if not isinstance(rows, list) or len(rows) < 76:
        return None
    now_ms = int(time.time() * 1000)
    candles = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 7:
            return None
        try:
            open_time = int(row[0])
            close_time = int(row[6])
            values = [float(row[i]) for i in (1, 2, 3, 4, 5)]
        except (ValueError, TypeError, IndexError):
            return None
        if close_time >= now_ms:
            continue
        if not all(math.isfinite(x) for x in values):
            return None
        op, hi, lo, cl, vol = values
        if min(op, hi, lo, cl) <= 0 or vol < 0 or hi < max(op, cl) or lo > min(op, cl):
            return None
        candles.append((open_time, op, hi, lo, cl, vol))
    df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if len(df) < 75:
        return None
    latest = int(df["time"].iloc[-1])
    if now_ms - (latest + INTERVAL_MS[interval]) > 2 * INTERVAL_MS[interval]:
        return None
    if (df["time"].diff().iloc[-60:] != INTERVAL_MS[interval]).any():
        return None
    return df


def rsi14(close):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    result = 100 - 100 / (1 + rs)
    result = result.mask((loss == 0) & (gain > 0), 100)
    return result.mask((loss == 0) & (gain == 0), 50)


def analyze(df):
    close, volume = df["close"], df["volume"]
    price = float(close.iloc[-1])
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    rsi = float(rsi14(close).iloc[-1])
    middle = close.rolling(20).mean()
    std = close.rolling(20).std(ddof=0)
    upper, lower = middle + 2 * std, middle - 2 * std
    width = (upper - lower) / middle.replace(0, float("nan"))
    avg_width = width.rolling(50).mean()
    band_range = float((upper - lower).iloc[-1])
    avg_vol = float(volume.shift(1).rolling(20).mean().iloc[-1])
    if not all(map(math.isfinite, (price, rsi, band_range, avg_vol))) or band_range <= 0:
        return None
    pct_b = (price - float(lower.iloc[-1])) / band_range
    vol_ratio = float(volume.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0
    squeeze = bool((width.iloc[-6:] < 0.8 * avg_width.iloc[-6:]).fillna(False).any())
    score = 0
    notes = []
    if price > float(ema20.iloc[-1]) > float(ema50.iloc[-1]):
        score += 1
        notes.append("EMA trend")
    if 50 <= rsi <= 70:
        score += 1
        notes.append("RSI")
    if rsi > 75:
        score -= 1
    if squeeze and 0.8 <= pct_b <= 1.1:
        score += 1
        notes.append("BB sıkışma/kırılım")
    if pct_b > 1.1:
        score -= 1
    if vol_ratio >= 2:
        score += 1
        notes.append("Hacim")
    bounce = rsi < 30 and pct_b <= 0.05 and vol_ratio >= 2
    return {"score": score, "bounce": bounce, "rsi": rsi, "price": price,
            "vol_ratio": vol_ratio, "notes": notes}


def levels_for(df):
    result = technical_analysis.technical_levels(df, candles_are_closed=True)
    if not isinstance(result, dict):
        return None
    try:
        support = float(result["support"])
        resistance = float(result["resistance"])
        stop = float(result["stop"])
        target = float(result["target1"])
        price = float(df["close"].iloc[-1])
        rr = (target - price) / (price - stop)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(map(math.isfinite, (support, resistance, stop, target, price, rr))):
        return None
    if not (0 < stop < price < target <= resistance and support < resistance and rr >= MIN_RR):
        return None
    return {"support": support, "resistance": resistance, "entry": price,
            "stop": stop, "target": target, "rr": rr}


def load_state():
    try:
        value = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Sinyal kayıt dosyası okunamadı: {exc}") from exc


def save_state(state):
    temporary = STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(STATE_FILE)


def send_telegram(message):
    if DRY_RUN:
        print("DRY_RUN: Bildirim gönderilmedi.\n" + message)
        return False
    if not TOKEN or not CHAT_ID:
        raise RuntimeError("Telegram token veya chat ID eksik")
    response = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": message}, timeout=20)
    response.raise_for_status()
    if not response.json().get("ok"):
        raise RuntimeError("Telegram mesajı kabul etmedi")
    return True


def main():
    state = load_state()
    symbols = get_symbols()
    print(f"{len(symbols)} Spot USDT paritesi incelenecek")
    sent = 0
    for symbol in symbols:
        try:
            one_hour = get_klines(symbol, "1h")
            four_hour = get_klines(symbol, "4h")
            if one_hour is None or four_hour is None:
                continue
            a1, a4 = analyze(one_hour), analyze(four_hour)
            if not a1 or not a4:
                continue
            total = a1["score"] + a4["score"]
            if total < MIN_SCORE and not a1["bounce"]:
                continue
            levels = levels_for(four_hour)
            if not levels:
                continue
            # 1h and 4h must both confirm; do not alert on bounce alone.
            if total < MIN_SCORE:
                continue
            candle_time = int(four_hour["time"].iloc[-1])
            signal_key = f"{symbol}:4h:{candle_time}"
            if signal_key in state:
                continue
            timestamp = datetime.fromtimestamp(candle_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            message = (
                f"SPOT | {symbol} | {timestamp}\n"
                f"Puan: {total}/8 | RSI(1s): {a1['rsi']:.1f} | RSI(4s): {a4['rsi']:.1f}\n"
                f"Giriş referansı: {levels['entry']:.8g}\n"
                f"Destek: {levels['support']:.8g} | Direnç: {levels['resistance']:.8g}\n"
                f"Stop: {levels['stop']:.8g} | Hedef: {levels['target']:.8g}\n"
                f"Risk/Ödül: {levels['rr']:.2f} | 1s hacim oranı: {a1['vol_ratio']:.2f}\n"
                "Koşul: 1s + 4s puan teyidi ve 4s risk/ödül filtresi.\n"
                "Analiz uyarısıdır; otomatik emir değildir."
            )
            if send_telegram(message):
                state[signal_key] = timestamp
                save_state(state)
                sent += 1
        except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError) as exc:
            print(f"{symbol}: atlandı ({exc})")
        time.sleep(0.08)
    print(f"Tarama tamamlandı. Yeni gönderilen sinyal: {sent}")


if __name__ == "__main__":
    main()

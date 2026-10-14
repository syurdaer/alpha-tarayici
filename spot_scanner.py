"""
Binance SPOT tarayıcı
- alpha_scanner.py ile aynı göstergeleri kullanır (EMA, RSI, Bollinger, hacim)
- 1 saat + 4 saat birlikte puanlanır (maksimum 8 puan)
- Hacmi yüksek USDT çiftlerini tarar, güçlü adayları Telegram'a gönderir
"""
import os
import time

import pandas as pd
import requests

from alpha_scanner import HEADERS, analyze, send_telegram, to_float

# ---------- AYARLAR ----------
MIN_QUOTE_VOLUME = float(os.getenv("SPOT_MIN_VOLUME", "5000000"))  # günlük min. hacim ($)
MAX_COINS = int(os.getenv("SPOT_MAX_COINS", "100"))                # taranacak en fazla coin
MIN_SCORE = int(os.getenv("SPOT_MIN_SCORE", "6"))                  # alarm eşiği (0-8)

# Sadece piyasa verisi veren, bölge engeli daha az olan resmi Binance adresi
BASE = "https://data-api.binance.vision/api/v3"

STABLES = {"USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDE", "EUR", "EURI",
           "AEUR", "USD1", "XUSD", "PYUSD", "UST"}


def get_symbols():
    r = requests.get(f"{BASE}/ticker/24hr", headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for t in r.json():
        s = t.get("symbol", "")
        if not s.endswith("USDT") or s[:-4] in STABLES:
            continue
        vol = to_float(t.get("quoteVolume"))
        if vol >= MIN_QUOTE_VOLUME:
            out.append({"symbol": s, "vol": vol})
    out.sort(key=lambda x: x["vol"], reverse=True)
    return out[:MAX_COINS]


def get_klines(symbol, interval, limit=120):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(f"{BASE}/klines", params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


def main():
    coins = get_symbols()
    print(f"{len(coins)} spot coin taranacak")

    hits, bounces = [], []
    for c in coins:
        try:
            a1h = analyze(get_klines(c["symbol"], "1h"))
            a4h = analyze(get_klines(c["symbol"], "4h"))
        except Exception as e:  # tek coin hatası tüm taramayı durdurmasın
            print(f"{c['symbol']} atlandı: {e}")
            continue
        finally:
            time.sleep(0.1)

        if not a1h or not a4h:
            continue
        total = a1h["score"] + a4h["score"]
        row = {"symbol": c["symbol"], "total": total, "a1h": a1h, "a4h": a4h}
        if total >= MIN_SCORE:
            hits.append(row)
        if a1h["bounce"]:
            bounces.append(row)

    if not hits and not bounces:
        print("Güçlü aday yok, mesaj gönderilmedi.")
        return

    hits.sort(key=lambda x: x["total"], reverse=True)
    lines = []
    if hits:
        lines.append("📈 SPOT - GÜÇLÜ ADAYLAR (puan /8)")
        for h in hits[:8]:
            lines.append(
                f"{h['symbol']}  {h['total']}/8  fiyat {h['a1h']['price']:.6g}\n"
                f"  1sa: {', '.join(h['a1h']['notes']) or '-'} (RSI {h['a1h']['rsi']:.0f})\n"
                f"  4sa: {', '.join(h['a4h']['notes']) or '-'} (RSI {h['a4h']['rsi']:.0f})"
            )
    if bounces:
        lines.append("\n↩️ SPOT - TEPKİ ADAYLARI (aşırı satım + alt bant + hacim)")
        for b in bounces[:5]:
            lines.append(f"{b['symbol']}  fiyat {b['a1h']['price']:.6g}  RSI {b['a1h']['rsi']:.0f}")
    lines.append("\n⚠️ Yatırım tavsiyesi değildir, grafiği kontrol et.")
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"HATA: {e}")
        raise SystemExit(1)

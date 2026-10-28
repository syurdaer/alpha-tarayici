"""
Binance Alpha tarayıcı
- Göstergeler: EMA20/EMA50, RSI14, Bollinger Band (20,2), hacim oranı
- 15 dakika + 1 saat birlikte puanlanır (maksimum 8 puan)
- Güçlü adayları Telegram'a gönderir
"""
import os
import sys
import time

import pandas as pd
import requests
import technical_analysis
# ---------- AYARLAR (GitHub Secrets / ortam değişkenleri ile değiştirilebilir) ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"  # 1 ise Telegram'a göndermez, ekrana yazar

MIN_VOLUME_24H = float(os.getenv("MIN_VOLUME_24H", "300000"))  # günlük min. hacim ($)
MIN_LIQUIDITY = float(os.getenv("MIN_LIQUIDITY", "100000"))    # min. likidite ($)
MAX_COINS = int(os.getenv("MAX_COINS", "50"))                  # taranacak en fazla coin
MIN_TOTAL_SCORE = int(os.getenv("MIN_TOTAL_SCORE", "6"))       # alarm eşiği (0-8)

BASE = "https://www.binance.com/bapi/defi/v1/public"
TOKEN_LIST_URL = f"{BASE}/wallet-direct/buw/wallet/cex/alpha/all/token/list"
KLINE_URL = f"{BASE}/alpha-trade/klines"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# ---------- VERİ ÇEKME ----------
def get_tokens():
    r = requests.get(TOKEN_LIST_URL, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json().get("data") or []
    tokens = []
    for t in data:
        alpha_id = t.get("alphaId")
        if not alpha_id or t.get("offline") or t.get("offsell"):
            continue
        vol = to_float(t.get("volume24h"))
        liq = to_float(t.get("liquidity"))
        if vol < MIN_VOLUME_24H or liq < MIN_LIQUIDITY:
            continue
        tokens.append({"id": alpha_id, "symbol": t.get("symbol", alpha_id), "vol": vol})
    tokens.sort(key=lambda x: x["vol"], reverse=True)
    return tokens[:MAX_COINS]


def get_klines(alpha_id, interval, limit=120):
    params = {"symbol": f"{alpha_id}USDT", "interval": interval, "limit": limit}
    r = requests.get(KLINE_URL, params=params, headers=HEADERS, timeout=20)
    r.raise_for_status()
    rows = r.json().get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().reset_index(drop=True)


# ---------- GÖSTERGELER ----------
def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def analyze(df):
    """Son KAPANMIŞ mumu puanlar. Yeterli veri yoksa None döner."""
    if df is None or len(df) < 75:
        return None

    c, v = df["close"], df["volume"]
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    rsi14 = rsi(c)

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    bandwidth = (upper - lower) / mid
    bw_avg = bandwidth.rolling(50).mean()
    vol_avg = v.rolling(20).mean().shift(1)

    i = -2  # -1 henüz kapanmamış mum, onu kullanmıyoruz
    price = c.iloc[i]
    band_range = upper.iloc[i] - lower.iloc[i]
    if band_range == 0 or pd.isna(rsi14.iloc[i]) or pd.isna(vol_avg.iloc[i]):
        return None

    pct_b = (price - lower.iloc[i]) / band_range  # 0 = alt bant, 1 = üst bant
    squeeze = bool((bandwidth.iloc[-7:-1] < bw_avg.iloc[-7:-1] * 0.8).any())
    vol_ratio = v.iloc[i] / vol_avg.iloc[i] if vol_avg.iloc[i] > 0 else 0
    r = float(rsi14.iloc[i])

    score, notes = 0, []
    if price > ema20.iloc[i] > ema50.iloc[i]:
        score += 1
        notes.append("trend↑")
    if 50 <= r <= 70:
        score += 1
        notes.append("RSI iyi")
    if r > 75:
        score -= 1
        notes.append("RSI aşırı alım")
    if squeeze and 0.8 <= pct_b <= 1.1:
        score += 1
        notes.append("BB sıkışma+kırılım")
    if pct_b > 1.1:
        score -= 1
        notes.append("BB dışına taştı")
    if vol_ratio >= 2:
        score += 1
        notes.append(f"hacim x{vol_ratio:.1f}")

    bounce = r < 30 and pct_b <= 0.05 and vol_ratio >= 2
    return {"score": score, "notes": notes, "rsi": r, "vol_ratio": vol_ratio,
            "price": float(price), "bounce": bounce}


# ---------- TELEGRAM ----------
def send_telegram(text):
    if DRY_RUN or not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("---- MESAJ (gönderilmedi) ----")
        print(text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)
    r.raise_for_status()


# ---------- ANA AKIŞ ----------
def main():
    tokens = get_tokens()
    print(f"{len(tokens)} coin taranacak")

    hits, bounces = [], []
    for t in tokens:
        try:
              try:
                df4h = get_klines(t["id"], "4h")
                levels4h = technical_analysis.technical_levels(df4h)
                print(f"{t['symbol']} 4h teknik seviyeler: {levels4h}")
              except Exception as e:
                levels4h = None
                print(f"{t['symbol']} 4h analiz atlandı: {e}")

            a15 = analyze(get_klines(t["id"], "15m"))
            a1h = analyze(get_klines(t["id"], "1h"))
        except Exception as e:  # tek coin hatası tüm taramayı durdurmasın
            print(f"{t['symbol']} atlandı: {e}")
            continue
        finally:
            time.sleep(0.2)  # API'yi yormamak için kısa bekleme

        if not a15 or not a1h:
            continue
        total = a15["score"] + a1h["score"]
        row = {"symbol": t["symbol"], "total": total, "a15": a15, "a1h": a1h}
        if total >= MIN_TOTAL_SCORE:
            hits.append(row)
        if a15["bounce"]:
            bounces.append(row)

    if not hits and not bounces:
        print("Güçlü aday yok, mesaj gönderilmedi.")
        return

    hits.sort(key=lambda x: x["total"], reverse=True)
    lines = []
    if hits:
        lines.append("🔥 GÜÇLÜ ADAYLAR (puan /8)")
        for h in hits[:8]:
            lines.append(
                f"{h['symbol']}  {h['total']}/8  fiyat {h['a15']['price']:.6g}\n"
                f"  15dk: {', '.join(h['a15']['notes']) or '-'} (RSI {h['a15']['rsi']:.0f})\n"
                f"  1sa: {', '.join(h['a1h']['notes']) or '-'} (RSI {h['a1h']['rsi']:.0f})"
            )
    if bounces:
        lines.append("\n↩️ TEPKİ ADAYLARI (aşırı satım + alt bant + hacim)")
        for b in bounces[:5]:
            lines.append(f"{b['symbol']}  fiyat {b['a15']['price']:.6g}  RSI {b['a15']['rsi']:.0f}")
    lines.append("\n⚠️ Yatırım tavsiyesi değildir, grafiği kontrol et.")
    send_telegram("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"HATA: {e}")
        sys.exit(1)

"""Binance Alpha scanner: closed-candle scoring and guarded Telegram alerts."""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import technical_analysis

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
DRY_RUN = os.getenv('DRY_RUN', '0') == '1'
MIN_VOLUME_24H = float(os.getenv('MIN_VOLUME_24H', '300000'))
MIN_LIQUIDITY = float(os.getenv('MIN_LIQUIDITY', '100000'))
MAX_COINS = int(os.getenv('MAX_COINS', '50'))
MIN_TOTAL_SCORE = int(os.getenv('MIN_TOTAL_SCORE', '6'))
MIN_RR = float(os.getenv('MIN_RR', '1.5'))
STATE_FILE = Path(os.getenv('ALPHA_ALERT_STATE_FILE', 'alpha_alert_state.json'))
BASE = 'https://www.binance.com/bapi/defi/v1/public'
TOKEN_LIST_URL = f'{BASE}/wallet-direct/buw/wallet/cex/alpha/all/token/list'
KLINE_URL = f'{BASE}/alpha-trade/klines'
HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
INTERVAL_MS = {'15m': 900_000, '1h': 3_600_000, '4h': 14_400_000, '1d': 86_400_000}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def to_float(value, default=None):
    try:
        result = float(value)
        return result if pd.notna(result) and abs(result) != float('inf') else default
    except (TypeError, ValueError, OverflowError):
        return default


def api_data(url, params=None):
    response = SESSION.get(url, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or 'data' not in payload:
        raise ValueError('Beklenmeyen API yanıtı')
    if payload.get('success') is False or payload.get('code') not in (None, '000000', 0, '0'):
        raise ValueError(f"API hatası: {payload.get('code')} {payload.get('message', '')}")
    return payload['data']


def get_tokens():
    raw = api_data(TOKEN_LIST_URL)
    if not isinstance(raw, list):
        raise ValueError('Token listesi beklenen biçimde değil')
    tokens = []
    for token in raw:
        if not isinstance(token, dict):
            continue
        alpha_id = token.get('alphaId')
        if not alpha_id or token.get('offline') or token.get('offsell'):
            continue
        volume = to_float(token.get('volume24h'))
        liquidity = to_float(token.get('liquidity'))
        # Missing liquidity is not equivalent to adequate liquidity.
        if volume is None or liquidity is None or volume < MIN_VOLUME_24H or liquidity < MIN_LIQUIDITY:
            continue
        tokens.append({'id': str(alpha_id), 'symbol': str(token.get('symbol') or alpha_id), 'vol': volume})
    tokens.sort(key=lambda item: item['vol'], reverse=True)
    return tokens[:MAX_COINS]


def get_klines(alpha_id, interval, limit=120):
    if interval not in INTERVAL_MS:
        raise ValueError(f'Geçersiz zaman aralığı: {interval}')
    rows = api_data(KLINE_URL, {'symbol': f'{alpha_id}USDT', 'interval': interval, 'limit': limit})
    if not isinstance(rows, list) or not rows:
        return None
    if any(not isinstance(row, (list, tuple)) or len(row) < 6 for row in rows):
        return None
    df = pd.DataFrame([row[:6] for row in rows], columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna().sort_values('time').drop_duplicates('time', keep='last').reset_index(drop=True)
    if df.empty:
        return None
    now_ms = int(time.time() * 1000)
    # Only complete candles. Reject future and stale data.
    df = df[(df['time'] + INTERVAL_MS[interval] <= now_ms) & (df['time'] > 0)].copy()
    if df.empty or now_ms - int(df['time'].iloc[-1] + INTERVAL_MS[interval]) > INTERVAL_MS[interval] * 2:
        return None
    if (df[['open', 'high', 'low', 'close']] <= 0).any().any() or (df['volume'] < 0).any():
        return None
    if ((df['high'] < df[['open', 'close', 'low']].max(axis=1)) | (df['low'] > df[['open', 'close']].min(axis=1))).any():
        return None
    return df.reset_index(drop=True)


def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    result = 100 - 100 / (1 + gain / loss.replace(0, float('nan')))
    return result.mask((loss == 0) & (gain > 0), 100).mask((loss == 0) & (gain == 0), 50)


def analyze(df):
    if df is None or len(df) < 75:
        return None
    c, v = df['close'], df['volume']
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    rsi14 = rsi(c)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    upper, lower = mid + 2 * sd, mid - 2 * sd
    bandwidth = (upper - lower) / mid.replace(0, float('nan'))
    bw_avg = bandwidth.rolling(50).mean()
    vol_avg = v.rolling(20).mean().shift(1)
    i = -1  # get_klines already removed the incomplete candle
    price = float(c.iloc[i])
    band_range = upper.iloc[i] - lower.iloc[i]
    if pd.isna(band_range) or band_range <= 0 or pd.isna(rsi14.iloc[i]) or pd.isna(vol_avg.iloc[i]):
        return None
    pct_b = float((price - lower.iloc[i]) / band_range)
    squeeze = bool((bandwidth.iloc[-6:] < bw_avg.iloc[-6:] * 0.8).fillna(False).any())
    vol_ratio = float(v.iloc[i] / vol_avg.iloc[i]) if vol_avg.iloc[i] > 0 else 0.0
    r = float(rsi14.iloc[i])
    score, notes = 0, []
    if price > ema20.iloc[i] > ema50.iloc[i]:
        score += 1
        notes.append('trend↑')
    if 50 <= r <= 70:
        score += 1
        notes.append('RSI uygun')
    if r > 75:
        score -= 1
        notes.append('RSI aşırı alım')
    if squeeze and 0.8 <= pct_b <= 1.1:
        score += 1
        notes.append('BB sıkışma+kırılım adayı')
    if pct_b > 1.1:
        score -= 1
        notes.append('BB aşırı uzama')
    if vol_ratio >= 2:
        score += 1
        notes.append(f'hacim x{vol_ratio:.1f}')
    bounce = r < 30 and pct_b <= 0.05 and vol_ratio >= 2
    return {'score': score, 'notes': notes, 'rsi': r, 'vol_ratio': vol_ratio,
            'price': price, 'bounce': bounce, 'time': int(df['time'].iloc[-1])}


def validated_levels(df):
    if df is None or len(df) < 60:
        return None
    levels = technical_analysis.technical_levels(df, candles_are_closed=True)
    if not isinstance(levels, dict):
        return None
    keys = ('support', 'resistance', 'stop', 'target1', 'risk_reward')
    if any(to_float(levels.get(key)) is None for key in keys):
        return None
    support, resistance, stop, target, rr = [float(levels[key]) for key in keys]
    entry = float(df['close'].iloc[-1])
    if not (0 < stop < entry < target <= resistance and support < resistance and rr >= MIN_RR):
        return None
    actual_rr = (target - entry) / (entry - stop)
    if actual_rr < MIN_RR or abs(actual_rr - rr) > max(0.25, actual_rr * 0.15):
        return None
    return {**levels, 'entry_reference': entry}


def load_state():
    try:
        data = json.loads(STATE_FILE.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp = STATE_FILE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(STATE_FILE)


def send_telegram(message):
    if DRY_RUN:
        print('DRY_RUN — Telegram gönderilmedi:\n' + message)
        return False
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print('Telegram kimlik bilgileri eksik; gönderim yapılmadı.')
        return False
    response = requests.post(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
                             data={'chat_id': TELEGRAM_CHAT_ID, 'text': message}, timeout=20)
    response.raise_for_status()
    if response.json().get('ok') is not True:
        raise ValueError('Telegram gönderimi onaylamadı')
    return True


def main():
    tokens = get_tokens()
    print(f'{len(tokens)} Alpha token taranacak')
    state = load_state()
    candidates = []
    for token in tokens:
        try:
            df4h = get_klines(token['id'], '4h')
            df15 = get_klines(token['id'], '15m')
            df1h = get_klines(token['id'], '1h')
            a15, a1h = analyze(df15), analyze(df1h)
            levels = validated_levels(df4h)
            if not a15 or not a1h or not levels:
                print(f"{token['symbol']}: eksik veri veya geçersiz risk/ödül")
                continue
            total = a15['score'] + a1h['score']
            bounce = a15['bounce']
            print(f"{token['symbol']}: 15m={a15['score']} 1h={a1h['score']} toplam={total} RR={levels['risk_reward']:.2f}")
            if total < MIN_TOTAL_SCORE and not bounce:
                continue
            # Use the same closed 15m candle timestamp to avoid duplicate alerts.
            kind = 'TEPKİ ADAYI' if bounce and total < MIN_TOTAL_SCORE else 'GÜÇLÜ ADAY'
            key = f"{token['id']}:{kind}"
            if state.get(key) == a15['time']:
                continue
            candidates.append((total, token, a15, a1h, levels, key, kind))
        except Exception as exc:
            print(f"{token['symbol']} atlandı: {exc}")
        finally:
            time.sleep(0.2)
    if not candidates:
        print('Yeni doğrulanmış aday yok; bildirim gönderilmedi.')
        return
    candidates.sort(key=lambda item: item[0], reverse=True)
    for total, token, a15, a1h, levels, key, kind in candidates[:8]:
        candle_time = datetime.fromtimestamp(a15['time'] / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        message = (f"Binance Alpha | {token['symbol']} | {kind}\n"
                   f"Mum başlangıcı: {candle_time}\n"
                   f"15m fiyat: {a15['price']:.8g} USDT | Puan: {total}/8\n"
                   f"15m RSI: {a15['rsi']:.1f} | Hacim x{a15['vol_ratio']:.1f}\n"
                   f"Destek: {float(levels['support']):.8g} | Direnç: {float(levels['resistance']):.8g}\n"
                   f"4h referans giriş: {float(levels['entry_reference']):.8g} USDT\n"
                   f"Stop: {float(levels['stop']):.8g} | Hedef: {float(levels['target1']):.8g}\n"
                   f"R/R: {float(levels['risk_reward']):.2f}\n"
                   f"Tetik: {', '.join(a15['notes']) or 'aşırı satım tepki adayı'}\n"
                   'Otomatik emir değildir; işlem öncesi grafiği doğrula.')
        if send_telegram(message):
            state[key] = a15['time']
            save_state(state)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'HATA: {exc}', file=sys.stderr)
        sys.exit(1)


import requests

URL = "https://www.binance.com/bapi/defi/v1/public/alpha-trade/klines"

tests = [
    ("XDP", "ALPHA_1214"),
    ("PONS", "ALPHA_1119"),
    ("AI", "ALPHA_1130"),
    ("AXIS", "ALPHA_1248"),
    ("CT", "ALPHA_1228"),
]

for name, alpha_id in tests:
    print(f"\n{name} ({alpha_id})")

    candidates = [
        f"{alpha_id}USDT",
        f"{alpha_id}USDC",
        f"{name}USDT",
        f"{name}USDC",
    ]

    for symbol in candidates:
        try:
            response = requests.get(
                URL,
                params={
                    "symbol": symbol,
                    "interval": "15m",
                    "limit": 5,
                },
                timeout=15,
            )
            data = response.json()

            rows = data.get("data")
            valid = (
                response.ok
                and data.get("success") is not False
                and isinstance(rows, list)
                and len(rows) > 0
            )

            print(
                f"  {symbol}: "
                f"{'VERI VAR' if valid else 'VERI YOK'} "
                f"| code={data.get('code')}"
            )
        except Exception as exc:
            print(f"  {symbol}: HATA - {exc}")

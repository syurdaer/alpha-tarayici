
"""
Technical Analysis V2 - Automated Tests
Run: python -m unittest test_technical_analysis.py
"""

import unittest

import pandas as pd

from technical_analysis import technical_levels


def make_candles(count=120):
    """Create deterministic sample OHLCV data."""
    rows = []

    for i in range(count):
        price = 100 + (i % 20) * 0.1

        rows.append({
            "time": i,
            "open": price - 0.2,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": 1000 + i,
        })

    return pd.DataFrame(rows)


class TestTechnicalAnalysis(unittest.TestCase):

    def setUp(self):
        self.df = make_candles()

    def test_valid_result(self):
        result = technical_levels(self.df)

        self.assertIsNotNone(result)

        for key in (
            "price",
            "support",
            "resistance",
            "atr",
            "signal",
            "entry",
            "stop",
            "target1",
            "target2",
            "risk_reward",
        ):
            self.assertIn(key, result)

    def test_insufficient_data(self):
        df = make_candles(20)

        self.assertIsNone(technical_levels(df))

    def test_positive_atr(self):
        result = technical_levels(self.df)

        self.assertGreater(result["atr"], 0)

    def test_support_below_resistance(self):
        result = technical_levels(self.df)

        self.assertLess(
            result["support"],
            result["resistance"]
        )

    def test_valid_long_setup(self):
        result = technical_levels(self.df)

        self.assertIsNotNone(result["entry"])
        self.assertLess(
            result["stop"],
            result["entry"]
        )
        self.assertGreater(
            result["target1"],
            result["entry"]
        )
        self.assertGreaterEqual(
            result["target2"],
            result["target1"]
        )
        self.assertGreater(
            result["risk_reward"],
            0
        )

    def test_unfinished_candle_excluded(self):
        df = self.df.copy()

        original = technical_levels(df)

        # Change only the unfinished final candle.
        df.loc[df.index[-1], "high"] = 10000
        df.loc[df.index[-1], "low"] = 0.01
        df.loc[df.index[-1], "close"] = 9000

        modified = technical_levels(df)

        for key in (
            "price",
            "support",
            "resistance",
            "atr",
        ):
            self.assertAlmostEqual(
                original[key],
                modified[key],
                places=8
            )

    def test_outside_range(self):
        df = self.df.copy()

        # Make the last completed candle close
        # above the historical resistance.
        idx = df.index[-2]
        df.loc[idx, "close"] = 200
        df.loc[idx, "high"] = 201

        result = technical_levels(df)

        self.assertEqual(
            result["signal"],
            "OUTSIDE_RANGE"
        )
        self.assertIsNone(result["entry"])
        self.assertIsNone(result["stop"])


if __name__ == "__main__":
    unittest.main()

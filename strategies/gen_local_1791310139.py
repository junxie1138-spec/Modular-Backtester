from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeWaveParams:
    concordance_threshold: float = 0.5
    trail_atr: float = 2.5


class GeneratedStrategy(BaseStrategy[RangeWaveParams]):
    strategy_id = "gen_local_1791310139"

    @classmethod
    def params_type(cls):
        return RangeWaveParams

    @staticmethod
    def warmup_bars(params: RangeWaveParams) -> int:
        return 32

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeWaveParams) -> pd.DataFrame:
        if not (0.0 < params.concordance_threshold <= 1.0):
            raise ValueError("concordance_threshold must be in (0, 1]")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0.0:
            raise ValueError("trail_atr must be positive and finite")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        denominator = out["previous_close"].where(out["previous_close"] > 0.0)
        out["normalized_range"] = ((data["high"] - data["low"]) / denominator).replace([np.inf, -np.inf], np.nan)
        out["range_change"] = out["normalized_range"].diff()
        out["range_direction"] = np.sign(out["range_change"])
        out["concordance"] = out["range_direction"] * out["range_direction"].shift(1)
        out["recent_concordance"] = out["concordance"].rolling(8, min_periods=8).mean()
        out["baseline_concordance"] = out["concordance"].shift(8).rolling(20, min_periods=20).mean()
        out["range_growth"] = out["normalized_range"].diff(8)
        out["price_advance"] = data["close"].diff(8)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["entry_event"] = (
            (out["recent_concordance"] >= params.concordance_threshold)
            & (out["recent_concordance"].shift(1) < params.concordance_threshold)
            & (out["baseline_concordance"] <= -0.1)
            & (out["range_growth"] > 0.0)
            & (out["price_advance"] > 0.0)
            & (out["atr"] > 0.0)
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeWaveParams,
    ) -> SignalFrame:
        n = len(data)
        raw = np.zeros(n, dtype=int)
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        events = indicators["entry_event"].fillna(False).to_numpy(dtype=bool)
        active = False
        pending = False
        armed = False
        entry = peak = stop = 0.0
        entry_bar = 0
        for i in range(n):
            opening, high, low, close = prices[i]
            valid = np.isfinite(prices[i]).all() and (prices[i] > 0.0).all()
            if not valid or not np.isfinite(atr[i]) or atr[i] <= 0.0:
                active = pending = armed = False
                continue
            if pending:
                # The prior close requested entry; this bar supplies the fill open.
                pending = False
                active = True
                armed = False
                entry = peak = opening
                entry_bar = i
                stop = entry - params.trail_atr * atr[i - 1]
            if active:
                # Test the stop established before this bar, avoiding OHLC ordering assumptions.
                # A breach requests liquidation at the following open.
                if low <= stop or i - entry_bar + 1 >= 18:
                    active = False
                    armed = False
                    continue
                peak = max(peak, high)
                if high >= entry * 1.02:
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                raw[i] = 1
            elif events[i]:
                pending = True
                raw[i] = 1
        df = pd.DataFrame({"signal": raw, "size": np.full(n, 0.95)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": [100, 100, 102, 104, 104, 103, 103],
        "high": [101, 103, 105, 106, 105, 104, 104],
        "low": [99, 99, 101, 103, 102, 102, 102],
        "close": [100, 102, 104, 105, 103, 103, 103],
        "volume": [1000] * 7,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0],
        "entry_event": [True, False, False, False, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, RangeWaveParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert (result.data["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    volume_multiple: float = 1.4
    trail_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790882085"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return 22

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if not np.isfinite(params.volume_multiple) or params.volume_multiple < 1.0:
            raise ValueError("volume_multiple must be finite and at least 1")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(20, min_periods=20).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["box_high"] = data["high"].shift(1).rolling(5, min_periods=5).max()
        out["box_low"] = data["low"].shift(1).rolling(5, min_periods=5).min()
        out["bull_volume"] = data["volume"].where(data["close"] > data["open"], 0.0).where(data["volume"].notna())
        out["box_volume"] = data["volume"].shift(1).rolling(5, min_periods=5).sum()
        out["box_bull_volume"] = out["bull_volume"].shift(1).rolling(5, min_periods=5).sum()
        out["bull_share"] = out["box_bull_volume"] / out["box_volume"].where(out["box_volume"] > 0)
        out["mean_volume"] = data["volume"].shift(1).rolling(20, min_periods=20).mean()
        out["volume_ratio"] = data["volume"] / out["mean_volume"].where(out["mean_volume"] > 0)
        out["box_atr"] = (out["box_high"] - out["box_low"]) / out["previous_atr"].where(out["previous_atr"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["box_atr"] > 0)
            & (out["box_atr"] <= 2.0)
            & (out["bull_share"] >= 0.65)
            & (out["volume_ratio"] >= params.volume_multiple)
            & (data["open"] <= out["box_high"])
            & (data["close"] > out["box_high"])
            & (data["close"] > data["open"])
            & (out["atr"] > 0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        previous_atrs = indicators["previous_atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        holding = False
        pending = False
        armed = False
        entry_price = stop = peak_close = 0.0
        entry_bar = -1
        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(previous_atrs[i]) and previous_atrs[i] > 0:
                    holding = True
                    armed = False
                    entry_price = opens[i]
                    entry_bar = i
                    peak_close = entry_price
                    stop = entry_price - params.trail_atr * previous_atrs[i]
            if holding:
                # A stop breach decides an exit at this close; execution is next open.
                # Test the existing stop before raising it to avoid intrabar ordering assumptions.
                breached = np.isfinite(lows[i]) and lows[i] <= stop
                expired = i - entry_bar + 1 >= 10
                if breached or expired:
                    holding = False
                    continue
                if np.isfinite(closes[i]) and closes[i] > 0:
                    peak_close = max(peak_close, closes[i])
                    if closes[i] >= entry_price * 1.01:
                        armed = True
                    if armed:
                        stop = max(stop, entry_price)
                        if np.isfinite(atrs[i]) and atrs[i] > 0:
                            stop = max(stop, peak_close - params.trail_atr * atrs[i])
                signals[i] = 1
            elif entries[i]:
                pending = True
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({
        "open": [100.0, 100.0, 101.2, 104.0, 103.0],
        "high": [100.5, 101.5, 104.5, 104.2, 103.5],
        "low": [99.5, 99.5, 100.5, 101.0, 102.5],
        "close": [100.0, 101.2, 104.0, 103.0, 103.0],
        "volume": [100.0] * 5,
    }, index=index)
    indicators = pd.DataFrame({
        "atr": [1.0] * 5,
        "previous_atr": [1.0] * 5,
        "entry": [True, False, False, False, False],
    }, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, StrategyParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    assert result.index.equals(data.index) and (result["size"] > 0).all()
    warm = strategy.indicators(data, StrategyParams())
    assert not warm["entry"].any()


if __name__ == "__main__":
    _self_check()

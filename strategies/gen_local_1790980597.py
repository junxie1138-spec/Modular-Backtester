from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SweepParams:
    atr_bars: int = 14
    sweep_fraction: float = 0.08
    overlap_edge: float = 0.35
    breakeven_pct: float = 0.005
    trail_atr: float = 0.8
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SweepParams]):
    strategy_id = "gen_local_1790980597"

    @classmethod
    def params_type(cls):
        return SweepParams

    def warmup_bars(self, params: SweepParams) -> int:
        return max(params.atr_bars + 1, 3)

    def indicators(self, data: pd.DataFrame, params: SweepParams) -> pd.DataFrame:
        if params.atr_bars < 1 or params.max_hold_bars not in (1, 2):
            raise ValueError("Invalid lookback or holding period")
        if not (params.sweep_fraction > 0 and 0 < params.overlap_edge < 0.5
                and params.breakeven_pct > 0 and params.trail_atr > 0):
            raise ValueError("Invalid thresholds")
        out = pd.DataFrame(index=data.index)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["previous_close"] = data["close"].shift(1)
        out["range"] = data["high"] - data["low"]
        out["previous_range"] = out["range"].shift(1)
        denominator = out["previous_range"].where(out["previous_range"] > 0)
        out["overlap_low"] = pd.concat([data["low"], out["previous_low"]], axis=1).max(axis=1)
        out["overlap_high"] = pd.concat([data["high"], out["previous_high"]], axis=1).min(axis=1)
        out["overlap_width"] = out["overlap_high"] - out["overlap_low"]
        width = out["overlap_width"].where(out["overlap_width"] > 0)
        out["overlap_position"] = (data["close"] - out["overlap_low"]) / width
        out["up_sweep"] = (data["high"] - out["previous_high"]) / denominator
        out["down_sweep"] = (out["previous_low"] - data["low"]) / denominator
        out["true_range"] = pd.concat(
            [out["range"], (data["high"] - out["previous_close"]).abs(),
             (data["low"] - out["previous_close"]).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        out["inside_overlap"] = data["close"].between(out["overlap_low"], out["overlap_high"])
        out["long_setup"] = (
            out["inside_overlap"] & (out["down_sweep"] >= params.sweep_fraction)
            & (out["overlap_position"] >= 1 - params.overlap_edge)
        )
        out["short_setup"] = (
            out["inside_overlap"] & (out["up_sweep"] >= params.sweep_fraction)
            & (out["overlap_position"] <= params.overlap_edge)
        )
        out["setup"] = np.select([out["long_setup"], out["short_setup"]], [1, -1], default=0)
        out["confirmed"] = out["setup"].where(out["setup"].eq(out["setup"].shift(1)), 0).astype(int)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: SweepParams,
    ) -> SignalFrame:
        desired = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        entry, stop, best = np.nan, np.nan, np.nan
        armed = False
        bars = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        confirmed = indicators["confirmed"].to_numpy(dtype=int)
        for i in range(self.warmup_bars(params) - 1, len(data)):
            op, high, low, close = bars[i]
            if not np.isfinite(bars[i]).all() or min(op, low, close) <= 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
                position = 0
                entry = np.nan
                continue
            if position:
                if not np.isfinite(entry):
                    # The previous close submitted entry; this bar supplies its opening reference.
                    entry = position * op
                    best = entry
                    stop = entry - params.trail_atr * atr[i - 1]
                    age, armed = 0, False
                age += 1
                # Signed prices make favorable stops increase for both longs and shorts.
                adverse = low if position == 1 else -high
                favorable = high if position == 1 else -low
                exited = adverse <= stop
                if not exited:
                    best = max(best, favorable)
                    armed = armed or best - entry >= abs(entry) * params.breakeven_pct
                    if armed:
                        stop = max(stop, entry, best - params.trail_atr * atr[i])
                        # A newly raised stop is checked against the close, avoiding assumed high/low ordering.
                        exited = position * close <= stop
                if exited or age >= params.max_hold_bars:
                    position = 0
                    entry = np.nan
            elif confirmed[i]:
                position = int(confirmed[i])
                entry = np.nan
            desired[i] = position
        df = pd.DataFrame({"signal": desired, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = SweepParams(atr_bars=2)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    data = pd.DataFrame(
        {"open": [105, 105, 106, 106, 106.5, 107],
         "high": [110, 109, 108, 107, 108, 108],
         "low": [100, 98, 96, 105, 106.25, 106],
         "close": [105, 106, 106, 106.5, 107, 107],
         "volume": [1000] * 6},
        index=pd.date_range("2020-01-01", periods=6),
    )
    for direction in (1, -1):
        sample = data.copy()
        if direction == -1:
            sample[["open", "close"]] = 210 - data[["open", "close"]]
            sample["high"], sample["low"] = 210 - data["low"], 210 - data["high"]
        result = strategy.generate_signals(sample, strategy.indicators(sample, params), ctx, params).data
        assert result["signal"].tolist() == [0, 0, 0, direction, direction, 0]
        assert (result["size"] > 0).all()
        for n in range(1, len(sample) + 1):
            prefix = sample.iloc[:n]
            check = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
            assert check.equals(result.iloc[:n])
        stopped = sample.copy()
        stopped.iloc[3, stopped.columns.get_loc("close")] = 105.9 if direction == 1 else 104.1
        check = strategy.generate_signals(stopped, strategy.indicators(stopped, params), ctx, params).data
        assert check["signal"].tolist() == [0, 0, 0, direction, 0, 0]


if __name__ == "__main__":
    _self_check()

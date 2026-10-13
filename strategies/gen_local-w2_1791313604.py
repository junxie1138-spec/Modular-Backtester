from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapShockParams:
    rank_window: int = 60
    quiet_window: int = 3
    atr_window: int = 14
    shock_percentile: float = 0.90
    quiet_percentile: float = 0.35
    retention: float = 0.75
    hold_bars: int = 4
    stop_atr: float = 1.8


class GeneratedStrategy(BaseStrategy[GapShockParams]):
    strategy_id = "gen_local-w2_1791313604"

    @classmethod
    def params_type(cls):
        return GapShockParams

    def warmup_bars(self, params: GapShockParams) -> int:
        if min(params.rank_window, params.quiet_window, params.atr_window) < 1:
            raise ValueError("Indicator windows must be positive")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not (0 < params.shock_percentile <= 1 and 0 <= params.quiet_percentile <= 1):
            raise ValueError("Invalid percentile thresholds")
        if not (0 < params.retention <= 1 and np.isfinite(params.stop_atr) and params.stop_atr > 0):
            raise ValueError("Invalid retention or stop multiplier")
        return int(max(200, params.atr_window + params.quiet_window + params.rank_window + 2))

    def indicators(self, data: pd.DataFrame, params: GapShockParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["previous_high"] = data["high"].shift(1)
        out["previous_low"] = data["low"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["gap"] = data["open"] - out["previous_close"]
        out["gap_strength"] = out["gap"].abs() / out["previous_atr"].where(out["previous_atr"] > 0)
        out["gap_rank"] = out["gap_strength"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["quiet_activity"] = out["gap_strength"].rolling(params.quiet_window, min_periods=params.quiet_window).mean().shift(1)
        out["quiet_rank"] = out["quiet_activity"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["retained_gap"] = (data["close"] - out["previous_close"]) / out["gap"].where(out["gap"] != 0)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapShockParams) -> SignalFrame:
        warmup = self.warmup_bars(params)
        ind = indicators
        needed = ["gap_rank", "quiet_rank", "retained_gap", "ma200", "atr", "gap", "previous_high", "previous_low"]
        ready = pd.Series(np.isfinite(ind[needed].to_numpy(dtype=float)).all(axis=1), index=data.index)
        ready &= np.isfinite(data["open"]) & np.isfinite(data["close"])
        ready &= (data["open"] > 0) & (data["close"] > 0) & (ind["atr"] > 0)
        ready &= np.arange(len(data)) >= warmup - 1
        event = ready & (ind["gap_rank"] >= params.shock_percentile)
        event &= (ind["quiet_rank"] <= params.quiet_percentile) & (ind["retained_gap"] >= params.retention)
        long_entry = event & (ind["gap"] > 0) & (data["open"] > ind["previous_high"]) & (data["close"] > ind["ma200"])
        short_entry = event & (ind["gap"] < 0) & (data["open"] < ind["previous_low"]) & (data["close"] < ind["ma200"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atr = ind["atr"].to_numpy(dtype=float)
        raw = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        distance, stop = 0.0, np.nan
        for i in range(len(data)):
            if side:
                if entry_bar < 0:
                    # Resolve the pending entry only when its fill bar arrives.
                    entry_bar = i
                    stop = opens[i] - side * distance
                stopped = not np.isfinite(closes[i]) or not np.isfinite(stop) or side * (closes[i] - stop) < 0
                expired = i - entry_bar >= params.hold_bars - 1
                if stopped or expired:
                    side, entry_bar = 0, -1
            elif candidates[i]:
                side, entry_bar = int(candidates[i]), -1
                # Freeze the signal-bar ATR; the resulting stop never trails.
                distance = params.stop_atr * atr[i]
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapShockParams()
    n = 207
    index = pd.date_range("2020-01-01", periods=n, freq="D")
    for direction, stopped in [(1, False), (1, True), (-1, True)]:
        data = pd.DataFrame({"open": 100.0, "high": 110.0, "low": 90.0, "close": 100.0 + direction, "volume": 1000.0}, index=index)
        ind = pd.DataFrame({"gap_rank": 0.0, "quiet_rank": 0.0, "retained_gap": 1.0, "ma200": 100.0, "atr": 1.0, "gap": float(direction), "previous_high": 99.0, "previous_low": 101.0}, index=index)
        ind.iloc[199, ind.columns.get_loc("gap_rank")] = 1.0
        if stopped:
            data.iloc[200, data.columns.get_loc("close")] = 100.0 + 4 * direction
            data.iloc[201, data.columns.get_loc("close")] = 100.0 - direction
            data.iloc[202, data.columns.get_loc("close")] = 100.0 - 3 * direction
        result = strategy.generate_signals(data, ind, None, params).data
        expected = [0] * n
        expected[200:204 if not stopped else 203] = [direction] * (4 if not stopped else 3)
        assert result["signal"].tolist() == expected
        assert (result["size"] > 0).all()
    assert strategy.indicators(data, params).index.equals(data.index)


if __name__ == "__main__":
    _self_check()

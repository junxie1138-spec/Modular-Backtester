from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapParticipationParams:
    recent_bars: int = 5
    baseline_bars: int = 20
    volume_multiple: float = 1.25
    body_atr: float = 0.45
    gap_shift_z: float = 0.75
    profit_target: float = 0.04
    max_hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapParticipationParams]):
    strategy_id = "gen_local-w2_1791326129"

    @classmethod
    def params_type(cls):
        return GapParticipationParams

    def warmup_bars(self, params):
        return int(params.baseline_bars + params.recent_bars + 1)

    def indicators(self, data, params):
        if (params.recent_bars < 2 or params.baseline_bars < 2
                or params.max_hold_bars < 1 or params.profit_target <= 0
                or params.volume_multiple <= 0 or params.body_atr <= 0
                or params.gap_shift_z <= 0):
            raise ValueError("Lookbacks must be >= 2 and thresholds positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).replace(0.0, np.nan)
        out["gap"] = data["open"] / previous_close - 1.0
        out["recent_gap"] = out["gap"].rolling(params.recent_bars).mean()
        old_gap = out["gap"].shift(params.recent_bars)
        out["baseline_gap"] = old_gap.rolling(params.baseline_bars).mean()
        out["baseline_gap_std"] = old_gap.rolling(params.baseline_bars).std()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["prior_atr"] = out["true_range"].rolling(params.baseline_bars).mean().shift(1)
        out["prior_volume"] = data["volume"].rolling(params.baseline_bars).mean().shift(1)
        out["volume_ratio"] = data["volume"] / out["prior_volume"].replace(0.0, np.nan)
        out["body_atr"] = (data["close"] - data["open"]) / out["prior_atr"].replace(0.0, np.nan)
        out["gap_changepoint"] = (
            (out["baseline_gap"] <= 0.0)
            & (out["recent_gap"] > 0.0)
            & (out["baseline_gap_std"] > 0.0)
            & ((out["recent_gap"] - out["baseline_gap"])
               > params.gap_shift_z * out["baseline_gap_std"])
        ).fillna(False)
        out["volume_move"] = (
            (out["volume_ratio"] >= params.volume_multiple)
            & (out["body_atr"] >= params.body_atr)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        entries = (indicators["gap_changepoint"].fillna(False)
                   & indicators["volume_move"].fillna(False)).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if entry_bar >= 0:
                if i == entry_bar:
                    # The previous close's decision fills at this bar's open.
                    entry_price = opens[i]
                target_hit = (
                    np.isfinite(entry_price) and entry_price > 0.0
                    and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                expired = i - entry_bar + 1 >= params.max_hold_bars
                if target_hit or expired or not np.isfinite(entry_price) or entry_price <= 0.0:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif entries[i]:
                signal[i] = 1
                entry_bar = i + 1
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapParticipationParams(max_hold_bars=3)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=26)
    gates = pd.DataFrame({"gap_changepoint": True, "volume_move": True}, index=index)
    result = strategy.generate_signals(data, gates, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 1, 1, 1]
    data.loc[index[2], "close"] = 105.0
    result = strategy.generate_signals(data, gates, ctx, params).data
    assert result["signal"].iloc[:4].tolist() == [0, 1, 1, 0]
    for column in gates.columns:
        blocked = gates.copy()
        blocked[column] = False
        assert not strategy.generate_signals(data, blocked, ctx, params).data["signal"].any()
    derived = strategy.indicators(data, params)
    assert not derived["gap_changepoint"].any()
    assert not derived["volume_move"].any()
    assert strategy.warmup_bars(params) == 26
    assert (result["size"] > 0.0).all()


if __name__ == "__main__":
    _self_check()

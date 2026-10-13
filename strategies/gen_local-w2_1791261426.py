from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    recent_bars: int = 5
    baseline_bars: int = 30
    gap_bars: int = 8
    downshift_z: float = 1.5
    contraction_ratio: float = 0.75
    gap_fraction: float = 0.75
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local-w2_1791261426"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return max(params.recent_bars + params.baseline_bars + 1, params.gap_bars + 1)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        if min(params.recent_bars, params.baseline_bars, params.gap_bars, params.max_hold) < 1:
            raise ValueError("Lookbacks and max_hold must be positive.")
        if not (0 < params.contraction_ratio < 1 and 0 < params.gap_fraction <= 1):
            raise ValueError("Invalid contraction_ratio or gap_fraction.")
        if not (np.isfinite(params.downshift_z) and params.downshift_z > 0
                and np.isfinite(params.profit_target) and params.profit_target > 0):
            raise ValueError("Thresholds must be finite and positive.")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1).where(lambda x: x > 0)
        out["normalized_range"] = ((data["high"] - data["low"]) / previous_close).where(
            data["high"] >= data["low"]
        ).replace([np.inf, -np.inf], np.nan)
        out["recent_range"] = out["normalized_range"].rolling(params.recent_bars).mean()
        baseline = out["normalized_range"].shift(params.recent_bars)
        out["baseline_mean"] = baseline.rolling(params.baseline_bars).mean()
        out["baseline_std"] = baseline.rolling(params.baseline_bars).std(ddof=0)
        error = (out["baseline_std"] / np.sqrt(params.recent_bars)).replace(0, np.nan)
        out["downshift_score"] = (
            (out["baseline_mean"] - out["recent_range"]) / error
        ).replace([np.inf, -np.inf], np.nan)
        out["range_downshift"] = (
            (out["downshift_score"] >= params.downshift_z)
            & (out["recent_range"] <= params.contraction_ratio * out["baseline_mean"])
        ).fillna(False)
        out["range_event"] = out["range_downshift"] & ~out["range_downshift"].shift(
            1, fill_value=False
        )
        out["overnight_gap"] = (data["open"] / previous_close - 1).where(
            data["open"] > 0
        ).replace([np.inf, -np.inf], np.nan)
        out["positive_gap_fraction"] = (
            out["overnight_gap"].gt(0).astype(float).where(out["overnight_gap"].notna())
            .rolling(params.gap_bars).mean()
        )
        out["mean_gap"] = out["overnight_gap"].rolling(params.gap_bars).mean()
        out["entry_trigger"] = (
            out["range_event"]
            & (out["positive_gap_fraction"] >= params.gap_fraction)
            & (out["mean_gap"] > 0)
        ).fillna(False)
        return out

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: GeneratedParams,
    ) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        entries = indicators["entry_trigger"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        active = False
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if active:
                # The pending decision fills at this bar's open, never a future open.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                target_hit = (
                    valid_entry and np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1 + params.profit_target)
                )
                timed_out = i - entry_bar + 1 >= params.max_hold
                if not valid_entry or target_hit or timed_out:
                    active = False
                else:
                    raw[i] = 1
            elif entries[i]:
                active = True
                entry_bar = i + 1
                raw[i] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = GeneratedParams()
    index = pd.date_range("2020-01-01", periods=22)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    triggers = pd.DataFrame({"entry_trigger": [True] + [False] * 21}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    signal = strategy.generate_signals(data, triggers, ctx, params).data["signal"]
    assert signal.tolist() == [0] + [1] * 18 + [0] * 3
    data.loc[index[1], "open"] = 102.0
    data.loc[index[1], "close"] = 104.1
    data.loc[index[2], "close"] = 107.0
    signal = strategy.generate_signals(data, triggers, ctx, params).data["signal"]
    assert signal.tolist() == [0, 1, 1] + [0] * 19
    indicators = strategy.indicators(data, params)
    assert not indicators["entry_trigger"].any()
    assert strategy.warmup_bars(params) == 36
    assert strategy.generate_signals(data.iloc[:0], triggers.iloc[:0], ctx, params).data.empty


if __name__ == "__main__":
    _self_check()

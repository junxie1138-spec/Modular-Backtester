from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEnergyParams:
    gap_window: int = 12
    atr_window: int = 14
    min_gaps: int = 3
    energy_threshold: float = 0.6
    stop_atr: float = 2.5
    hold_bars: int = 18
    min_size: float = 0.25
    max_size: float = 0.95

    def __post_init__(self):
        if min(self.gap_window, self.atr_window, self.min_gaps, self.hold_bars) < 1:
            raise ValueError("Windows, gap count and holding period must be positive.")
        if self.min_gaps > self.gap_window:
            raise ValueError("min_gaps cannot exceed gap_window.")
        if not (np.isfinite(self.energy_threshold) and self.energy_threshold > 0):
            raise ValueError("energy_threshold must be finite and positive.")
        if not (np.isfinite(self.stop_atr) and self.stop_atr > 0):
            raise ValueError("stop_atr must be finite and positive.")
        if not (0 < self.min_size <= self.max_size <= 1):
            raise ValueError("Position sizes must satisfy 0 < min_size <= max_size <= 1.")


class GeneratedStrategy(BaseStrategy[GapEnergyParams]):
    strategy_id = "gen_local_1791262931"

    @classmethod
    def params_type(cls):
        return GapEnergyParams

    def warmup_bars(self, params: GapEnergyParams) -> int:
        # ATR, its one-bar lag, and the subsequent gap-energy window.
        return int(params.atr_window + params.gap_window + 1)

    def indicators(self, data: pd.DataFrame, params: GapEnergyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1).where(out["previous_close"].notna())
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["previous_atr"] = out["atr"].shift(1).where(out["atr"].shift(1) > 0)
        out["gap"] = (data["open"] - out["previous_close"]) / out["previous_atr"]
        out["positive_gap"] = out["gap"].clip(lower=0)
        out["repayment"] = ((data["open"] - data["close"]) / out["previous_atr"]).clip(lower=0)
        out["retained_gap"] = (out["positive_gap"] - out["repayment"]).clip(lower=0)
        out["retained_energy"] = out["retained_gap"].pow(2)
        out["erased_energy"] = out["positive_gap"].pow(2) - out["retained_energy"]
        out["energy_balance"] = out["retained_energy"] - out["erased_energy"]
        # Integer rolling windows require a full window:
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
        out["score"] = out["energy_balance"].rolling(params.gap_window).sum()
        out["gap_count"] = out["gap"].gt(0).astype(float).where(out["gap"].notna()).rolling(params.gap_window).sum()
        out["momentum"] = data["close"] / data["close"].shift(params.gap_window).replace(0, np.nan) - 1
        out["strength_size"] = (params.max_size * out["score"] / (2 * params.energy_threshold)).clip(
            lower=params.min_size, upper=params.max_size
        )
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapEnergyParams,
    ) -> SignalFrame:
        eligible = (
            indicators["score"].ge(params.energy_threshold)
            & indicators["score"].shift(1).fillna(0).lt(params.energy_threshold)
            & indicators["gap_count"].ge(params.min_gaps)
            & indicators["momentum"].gt(0)
            & indicators["atr"].gt(0)
            & indicators["strength_size"].notna()
        )
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        sizes = indicators["strength_size"].to_numpy(dtype=float)
        entries = eligible.fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        decision_sizes = np.full(len(data), params.min_size, dtype=float)
        active = False
        entry_bar = -1
        entry_atr = 0.0
        fixed_stop = np.nan
        entry_size = params.min_size
        for i in range(len(data)):
            if active:
                if i == entry_bar:
                    # Observe the fill bar's open only when that bar arrives.
                    fixed_stop = opens[i] - params.stop_atr * entry_atr
                if (
                    not np.isfinite(fixed_stop)
                    or not np.isfinite(closes[i])
                    or closes[i] < fixed_stop
                    or i - entry_bar + 1 >= params.hold_bars
                ):
                    active = False
                    continue
            elif (
                i >= self.warmup_bars(params) - 1
                and entries[i]
                and np.isfinite(closes[i])
                and np.isfinite(atrs[i])
                and np.isfinite(sizes[i])
            ):
                active = True
                entry_bar = i + 1
                entry_atr = atrs[i]
                entry_size = sizes[i]
            if active:
                signals[i] = 1
                decision_sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": decision_sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = GapEnergyParams(gap_window=2, atr_window=2, min_gaps=1)
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                         "close": 100.0, "volume": 1000.0}, index=index)
    ind = strategy.indicators(data, params)
    ind["score"] = 0.0
    ind.loc[index[4], "score"] = params.energy_threshold
    ind["gap_count"], ind["momentum"], ind["atr"] = 1.0, 0.01, 2.0
    ind["strength_size"] = params.max_size * ind["score"] / (2 * params.energy_threshold)
    ind["strength_size"] = ind["strength_size"].clip(params.min_size, params.max_size)
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    held = strategy.generate_signals(data, ind, ctx, params).data
    assert held["signal"].iloc[:5].eq(0).all()
    assert held["signal"].iloc[5:23].eq(1).all() and held["signal"].iloc[23] == 0
    assert np.isclose(held["size"].iloc[5], params.max_size / 2)
    data.loc[index[5], ["open", "high", "low", "close"]] = [110.0, 111.0, 106.0, 107.0]
    data.loc[index[6], ["high", "low", "close"]] = [105.0, 99.0, 104.0]
    stopped = strategy.generate_signals(data, ind, ctx, params).data
    assert stopped["signal"].iloc[5:7].eq(1).all() and stopped["signal"].iloc[7] == 0
    prefix = strategy.generate_signals(data.iloc[:10], ind.iloc[:10], ctx, params).data
    pd.testing.assert_frame_equal(prefix, stopped.iloc[:10])
    ind.loc[index[4], ["score", "strength_size"]] = [2 * params.energy_threshold, params.max_size]
    stronger = strategy.generate_signals(data, ind, ctx, params).data
    assert stronger["size"].iloc[5] > stopped["size"].iloc[5]
    assert stronger["signal"].isin([0, 1]).all() and stronger["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()

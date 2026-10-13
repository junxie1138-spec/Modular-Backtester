from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapEnergyParams:
    energy_window: int = 20
    fast_share_min: float = 0.55
    gap_z_min: float = 1.25
    rejection_min: float = 0.40
    hold_bars: int = 8
    min_size: float = 0.25
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[GapEnergyParams]):
    strategy_id = "gen_local-w2_1791321520"

    @classmethod
    def params_type(cls):
        return GapEnergyParams

    def warmup_bars(self, params):
        return int(params.energy_window + 8)

    def indicators(self, data, params):
        if params.energy_window < 2 or params.hold_bars < 1:
            raise ValueError("Windows and holding period must be positive.")
        if not (0 < params.fast_share_min < 1 and params.gap_z_min > 0):
            raise ValueError("Invalid energy or gap threshold.")
        if not (0 < params.rejection_min <= 1):
            raise ValueError("Rejection threshold must be in (0, 1].")
        if not (0 < params.min_size <= params.max_size <= 1):
            raise ValueError("Sizes must satisfy 0 < min_size <= max_size <= 1.")
        out = pd.DataFrame(index=data.index)
        previous = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        closing = data["close"].where(lambda x: x > 0)
        out["previous_close"] = previous
        out["gap"] = np.log(opening / previous)
        out["intraday"] = np.log(closing / opening)
        out["gap_sigma"] = out["gap"].rolling(params.energy_window).std(ddof=0).shift(1)
        for half in (1, 2, 4):
            mean = out["gap"].rolling(half).mean()
            out[f"haar_{half}"] = np.sqrt(half / 2.0) * (mean - mean.shift(half))
            out[f"energy_{half}"] = out[f"haar_{half}"].pow(2).rolling(params.energy_window).mean()
        out["total_energy"] = out[["energy_1", "energy_2", "energy_4"]].sum(axis=1, min_count=3)
        out["fast_share"] = out["energy_1"] / out["total_energy"].where(out["total_energy"] > 0)
        out["gap_z"] = out["gap"].abs() / out["gap_sigma"].where(out["gap_sigma"] > 0)
        out["rejection"] = (-out["intraday"] / out["gap"].replace(0.0, np.nan)).clip(0.0, 1.0)
        out["strength"] = (
            (out["fast_share"] - params.fast_share_min) / (1.0 - params.fast_share_min)
        ).clip(0.0, 1.0) * (
            out["gap_z"] / (2.0 * params.gap_z_min)
        ).clip(0.0, 1.0) * out["rejection"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        eligible = (
            (indicators["fast_share"] >= params.fast_share_min)
            & (indicators["gap_z"] >= params.gap_z_min)
            & (indicators["rejection"] >= params.rejection_min)
            & indicators["strength"].notna()
            & indicators["gap"].notna()
        ).fillna(False).to_numpy(dtype=bool)
        direction = -np.sign(indicators["gap"].fillna(0.0)).to_numpy(dtype=int)
        strength = indicators["strength"].fillna(0.0).clip(0.0, 1.0).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), params.min_size, dtype=float)
        position, exit_bar, entry_size = 0, -1, params.min_size
        first = max(self.warmup_bars(params), ctx.warmup_bars) - 1
        for i in range(len(data)):
            if position:
                if i == exit_bar:
                    position = 0
                    continue
            elif i >= first and eligible[i] and direction[i] != 0:
                position = int(direction[i])
                exit_bar = i + params.hold_bars
                entry_size = params.min_size + (params.max_size - params.min_size) * strength[i]
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.min_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GapEnergyParams()
    strategy = GeneratedStrategy()
    n = strategy.warmup_bars(params) + params.hold_bars + 5
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    derived = strategy.indicators(data, params)
    assert derived.index.equals(index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    assert strategy.generate_signals(data, derived, ctx, params).data["signal"].eq(0).all()
    entry = strategy.warmup_bars(params)
    forced = pd.DataFrame(
        {"fast_share": 0.0, "gap_z": 0.0, "rejection": 0.0, "strength": 0.0, "gap": 0.0},
        index=index,
    )
    forced.iloc[entry] = [0.9, 3.0, 0.8, 0.6, -0.02]
    forced.iloc[entry + 2] = [0.9, 3.0, 0.8, 0.9, 0.02]
    result = strategy.generate_signals(data, forced, ctx, params).data
    expected = np.zeros(n, dtype=int)
    expected[entry + 1:entry + 1 + params.hold_bars] = 1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert np.allclose(
        result["size"].iloc[entry + 1:entry + 1 + params.hold_bars],
        params.min_size + (params.max_size - params.min_size) * 0.6,
    )
    forced.iloc[entry, forced.columns.get_loc("gap")] = 0.02
    result = strategy.generate_signals(data, forced, ctx, params).data
    expected[entry + 1:entry + 1 + params.hold_bars] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()

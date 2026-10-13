from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class FractionalSurpriseParams:
    frac_order: float = 0.45
    frac_window: int = 32
    score_window: int = 40
    vol_window: int = 20
    entry_z: float = 1.0
    target_vol: float = 0.12
    holding_bars: int = 4
    refractory_bars: int = 3
    spike_multiple: float = 2.5

    def __post_init__(self):
        if not 0.0 < self.frac_order < 1.0:
            raise ValueError("frac_order must lie between zero and one")
        if min(self.frac_window, self.score_window, self.vol_window) < 2:
            raise ValueError("lookbacks must be at least two")
        if self.holding_bars not in (3, 4, 5) or self.refractory_bars < 1:
            raise ValueError("holding_bars must be 3-5 and refractory_bars positive")
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.entry_z, self.target_vol, self.spike_multiple)):
            raise ValueError("thresholds and target_vol must be finite and positive")


class GeneratedStrategy(BaseStrategy[FractionalSurpriseParams]):
    strategy_id = "gen_local-w2_1791343374"

    @classmethod
    def params_type(cls):
        return FractionalSurpriseParams

    def warmup_bars(self, params: FractionalSurpriseParams) -> int:
        return int(max(params.frac_window + params.score_window + 2,
                       params.vol_window + params.refractory_bars + 2))

    def indicators(self, data: pd.DataFrame, params: FractionalSurpriseParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        weights = np.ones(params.frac_window)
        for k in range(1, params.frac_window):
            weights[k] = -weights[k - 1] * (params.frac_order - k + 1) / k
        # ponytail: fixed 32-bar default truncates fractional memory; extend frac_window if needed.
        out["fractional_close"] = sum(weights[k] * close.shift(k)
                                      for k in range(params.frac_window))
        out["fractional_mean"] = out["fractional_close"].rolling(params.score_window).mean().shift(1)
        out["fractional_std"] = out["fractional_close"].rolling(params.score_window).std().shift(1)
        out["z"] = ((out["fractional_close"] - out["fractional_mean"])
                    / out["fractional_std"].where(out["fractional_std"] > 0))
        out["return"] = close.pct_change(fill_method=None)
        out["daily_vol"] = out["return"].rolling(params.vol_window).std()
        out["annual_vol"] = out["daily_vol"] * np.sqrt(252.0)
        out["spike"] = (out["return"].abs()
                        > params.spike_multiple * out["daily_vol"].shift(1)).astype(int)
        out["cooldown"] = out["spike"].rolling(params.refractory_bars).max()
        out["entry"] = ((out["z"] > params.entry_z)
                        & (out["z"].shift(1) <= params.entry_z)
                        & (out["cooldown"] == 0)
                        & (out["annual_vol"] > 0))
        strength = (out["z"] / (2.0 * params.entry_z)).clip(0.0, 1.0)
        out["proposed_size"] = ((params.target_vol / out["annual_vol"].where(out["annual_vol"] > 0))
                                * strength).clip(0.05, 0.95)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: FractionalSurpriseParams) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.05}, index=data.index)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        sizes = indicators["proposed_size"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        held_sizes = np.full(len(data), 0.05, dtype=float)
        exit_at = -1
        entry_size = 0.05
        for i in range(len(data)):
            if i == exit_at:
                # Mandatory flat bar prevents an entry from cancelling the scheduled exit.
                continue
            if i > exit_at:
                if not entries[i] or not np.isfinite(sizes[i]) or sizes[i] <= 0:
                    continue
                exit_at = i + params.holding_bars
                entry_size = float(np.clip(sizes[i], 0.05, 0.95))
            signals[i] = 1
            held_sizes[i] = entry_size
        df["signal"] = signals
        df["size"] = held_sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = FractionalSurpriseParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=12, freq="D")
    data = pd.DataFrame({"close": 100.0}, index=index)
    forced = pd.DataFrame({"entry": True, "proposed_size": np.linspace(0.1, 0.9, 12)}, index=index)
    result = strategy.generate_signals(data, forced, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    assert result["size"].iloc[1:5].eq(forced["proposed_size"].iloc[0]).all()
    assert result["size"].gt(0).all()
    prices = pd.DataFrame({"close": 100.0 + np.arange(160) * 0.1
                          + np.sin(np.arange(160))}, index=pd.date_range("2020-01-01", periods=160))
    pd.testing.assert_frame_equal(strategy.indicators(prices.iloc[:120], params),
                                  strategy.indicators(prices, params).iloc[:120])

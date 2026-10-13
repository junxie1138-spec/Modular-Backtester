from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockParams:
    energy_window: int = 5
    rank_window: int = 63
    vol_window: int = 20
    shock_percentile: float = 0.90
    volume_percentile: float = 0.40
    target_vol: float = 0.12
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShockParams]):
    strategy_id = "gen_local-w2_1791292852"

    @classmethod
    def params_type(cls) -> type[ShockParams]:
        return ShockParams

    def warmup_bars(self, params: ShockParams) -> int:
        return int(max(params.energy_window + params.rank_window, params.vol_window + 1))

    def indicators(self, data: pd.DataFrame, params: ShockParams) -> pd.DataFrame:
        if min(params.energy_window, params.rank_window, params.vol_window) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        if not (0.5 < params.shock_percentile < 1.0 and 0.0 < params.volume_percentile < 0.5):
            raise ValueError("Invalid percentile thresholds")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0:
            raise ValueError("target_vol must be finite and positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return_energy"] = out["return"].pow(2)
        out["recent_energy"] = out["return_energy"].rolling(params.energy_window, min_periods=params.energy_window).sum()
        out["concentration"] = out["return_energy"] / out["recent_energy"].replace(0.0, np.nan)
        out["shock_rank"] = out["concentration"].rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        volume = data["volume"].where(data["volume"] > 0)
        out["volume_rank"] = volume.rolling(params.rank_window, min_periods=params.rank_window).rank(pct=True)
        out["volatility"] = out["return"].rolling(params.vol_window, min_periods=params.vol_window).std(ddof=0) * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShockParams) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five")
        valid = indicators[["return", "shock_rank", "volume_rank", "volatility"]].notna().all(axis=1)
        eligible = valid & (indicators["volatility"] > 0) & (indicators["shock_rank"] >= params.shock_percentile) & (indicators["volume_rank"] <= params.volume_percentile)
        candidates = np.where(eligible, -np.sign(indicators["return"].fillna(0.0)), 0).astype(int)
        volatility = indicators["volatility"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.95, dtype=float)
        position = 0
        exit_bar = -1
        entry_size = 0.95
        for i in range(len(data)):
            if position and i == exit_bar:
                position = 0
            elif not position and candidates[i]:
                position = int(candidates[i])
                exit_bar = i + params.hold_bars
                entry_size = float(np.clip(params.target_vol / volatility[i], 0.05, 0.95))
            signals[i] = position
            if position:
                sizes[i] = entry_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.95).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    derived = pd.DataFrame({"return": [0.01] * 5 + [-0.01] * 6, "shock_rank": 1.0, "volume_rank": 0.1, "volatility": [0.2] + [1.0] * 10}, index=index)
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, derived, None, ShockParams()).data
    assert result["signal"].tolist() == [0, -1, -1, -1, -1, 0, 1, 1, 1, 1, 0]
    assert np.allclose(result["size"].iloc[1:5], 0.6)
    assert np.allclose(result["size"].iloc[6:10], 0.12)
    assert (result["size"] > 0).all()
    warmup = strategy.indicators(data, ShockParams())
    flat = strategy.generate_signals(data, warmup, None, ShockParams()).data
    assert flat["signal"].eq(0).all()

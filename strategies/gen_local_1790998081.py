from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletParams:
    energy_window: int = 5
    migration_bars: int = 3
    risk_window: int = 20
    entry_threshold: float = 0.30
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[WaveletParams]):
    strategy_id = "gen_local_1790998081"

    @classmethod
    def params_type(cls) -> type[WaveletParams]:
        return WaveletParams

    def warmup_bars(self, params: WaveletParams) -> int:
        return int(max(16 + max(1, params.energy_window) + max(1, params.migration_bars), max(2, params.risk_window) + 1))

    def indicators(self, data: pd.DataFrame, params: WaveletParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        window = max(1, int(params.energy_window))
        migration_bars = max(1, int(params.migration_bars))
        risk_window = max(2, int(params.risk_window))
        log_close = np.log(data["close"].where(data["close"] > 0))
        returns = log_close.diff().replace([np.inf, -np.inf], np.nan)
        out["log_return"] = returns
        for scale in (2, 4, 8, 16):
            half = scale // 2
            block = returns.rolling(half, min_periods=half).mean()
            detail = (block - block.shift(half)) * np.sqrt(scale) / 2.0
            out[f"detail_{scale}"] = detail
            out[f"energy_{scale}"] = detail.pow(2).rolling(window, min_periods=window).mean()
            out[f"signed_energy_{scale}"] = (detail * detail.abs()).rolling(window, min_periods=window).mean()
        total = out[["energy_2", "energy_4", "energy_8", "energy_16"]].sum(axis=1, min_count=4)
        coarse = out["energy_8"] + out["energy_16"]
        out["total_energy"] = total
        out["coarse_energy"] = coarse
        out["coarse_share"] = coarse / total.clip(lower=1e-16)
        out["polarity"] = (out["signed_energy_8"] + out["signed_energy_16"]) / coarse.clip(lower=1e-16)
        out["energy_migration"] = out["coarse_share"].diff(migration_bars)
        out["annual_vol"] = returns.rolling(risk_window, min_periods=risk_window).std(ddof=0) * np.sqrt(252.0)
        required = ["polarity", "coarse_share", "energy_migration", "annual_vol"]
        out["ready"] = out[required].notna().all(axis=1) & total.gt(1e-16)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletParams) -> SignalFrame:
        threshold = max(0.0, float(params.entry_threshold))
        active = indicators["ready"].astype(bool) & indicators["coarse_share"].gt(0.55) & indicators["energy_migration"].gt(0.0)
        polarity = indicators["polarity"]
        candidates = np.select([active & polarity.gt(threshold), active & polarity.lt(-threshold)], [1, -1], default=0).astype(int)
        positions = np.zeros(len(data), dtype=int)
        state = 0
        for i, candidate in enumerate(candidates):
            candidate = int(candidate)
            if state == 0:
                state = candidate
            elif candidate == -state:
                state = 0
            positions[i] = state
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        allocation = (max(0.001, float(params.target_vol)) / indicators["annual_vol"].clip(lower=0.01)).clip(lower=0.05, upper=0.95)
        df["size"] = allocation.shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

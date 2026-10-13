from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VolumeRegimeParams:
    volume_window: int = 21
    pressure_window: int = 18
    efficiency_threshold: float = 0.35
    pressure_threshold: float = 0.20


class GeneratedStrategy(BaseStrategy[VolumeRegimeParams]):
    strategy_id = "gen_local_1790995855"

    @classmethod
    def params_type(cls) -> type[VolumeRegimeParams]:
        return VolumeRegimeParams

    def warmup_bars(self, params: VolumeRegimeParams) -> int:
        return max(2, params.volume_window) + max(2, params.pressure_window)

    def indicators(self, data: pd.DataFrame, params: VolumeRegimeParams) -> pd.DataFrame:
        v = max(2, params.volume_window)
        p = max(2, params.pressure_window)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).where(data["close"] > 0)
        volume = data["volume"].astype(float).where(data["volume"] >= 0)
        out["log_return"] = np.log(close).diff()
        out["volume_baseline"] = volume.rolling(v, min_periods=v).mean().shift(1)
        out["relative_volume"] = (volume / out["volume_baseline"].replace(0.0, np.nan)).clip(upper=3.0)
        out["signed_participation"] = np.sign(out["log_return"]) * out["relative_volume"]
        out["net_participation"] = out["signed_participation"].rolling(p, min_periods=p).sum()
        out["total_participation"] = out["relative_volume"].rolling(p, min_periods=p).sum()
        out["pressure"] = out["net_participation"] / out["total_participation"].replace(0.0, np.nan)
        out["net_return"] = out["log_return"].rolling(p, min_periods=p).sum()
        out["path_length"] = out["log_return"].abs().rolling(p, min_periods=p).sum()
        out["efficiency"] = (out["net_return"].abs() / out["path_length"].replace(0.0, np.nan)).clip(0.0, 1.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        valid = out["pressure"].notna() & out["efficiency"].notna()
        out["regime"] = np.where(valid, np.where(out["efficiency"] >= params.efficiency_threshold, 1, -1), 0)
        confirmed = valid & (out["pressure"].abs() >= max(0.0, params.pressure_threshold))
        out["entry"] = np.where(confirmed, np.sign(out["pressure"]) * out["regime"], 0).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VolumeRegimeParams,
    ) -> SignalFrame:
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, entry in enumerate(entries):
            if position == 0:
                position = int(entry)
            elif entry == -position:
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

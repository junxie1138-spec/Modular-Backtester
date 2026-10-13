from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class WaveletTensionParams:
    baseline_bars: int = 48
    energy_ceiling: float = 0.65
    tension_threshold: float = 0.20
    range_ceiling: float = 0.85


class GeneratedStrategy(BaseStrategy[WaveletTensionParams]):
    strategy_id = "gen_local-w2_1790991334"

    @classmethod
    def params_type(cls) -> type[WaveletTensionParams]:
        return WaveletTensionParams

    def warmup_bars(self, params: WaveletTensionParams) -> int:
        return max(2, int(params.baseline_bars)) + 11

    def indicators(self, data: pd.DataFrame, params: WaveletTensionParams) -> pd.DataFrame:
        baseline_bars = max(2, int(params.baseline_bars))
        out = pd.DataFrame(index=data.index)
        log_close = np.log(data["close"].where(data["close"] > 0))
        out["log_close"] = log_close
        for half in (1, 2, 4):
            recent = log_close.rolling(half, min_periods=half).mean()
            previous = log_close.shift(half).rolling(half, min_periods=half).mean()
            detail = np.sqrt(half / 2.0) * (recent - previous)
            energy = detail.pow(2).rolling(4, min_periods=4).mean()
            baseline = energy.shift(1).rolling(baseline_bars, min_periods=baseline_bars).mean()
            out[f"haar_{half}"] = detail
            out[f"energy_{half}"] = energy
            out[f"baseline_{half}"] = baseline
            out[f"ratio_{half}"] = energy / baseline.where(baseline > 0)
        daily_range = (data["high"] - data["low"]).where(data["high"] >= data["low"])
        out["daily_range"] = daily_range
        out["range_mean"] = daily_range.rolling(4, min_periods=4).mean()
        out["range_baseline"] = daily_range.shift(1).rolling(baseline_bars, min_periods=baseline_bars).mean()
        out["range_ratio"] = out["range_mean"] / out["range_baseline"].where(out["range_baseline"] > 0)
        midpoint = (data["high"] + data["low"]) / 2.0
        out["midpoint"] = midpoint
        out["center"] = midpoint.rolling(4, min_periods=4).mean()
        out["tension"] = (data["close"] - out["center"]) / out["range_mean"].where(out["range_mean"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: WaveletTensionParams) -> SignalFrame:
        ratios = indicators[["ratio_1", "ratio_2", "ratio_4"]]
        valid = ratios.notna().all(axis=1) & indicators[["range_ratio", "tension"]].notna().all(axis=1)
        compressed = valid & ratios.le(params.energy_ceiling).all(axis=1) & indicators["range_ratio"].le(params.range_ceiling)
        threshold = max(0.0, float(params.tension_threshold))
        raw = pd.Series(0, index=data.index, dtype=int)
        raw.loc[compressed & indicators["tension"].lt(-threshold)] = 1
        raw.loc[compressed & indicators["tension"].gt(threshold)] = -1
        confirmed = raw.where(raw.eq(raw.shift(1)), 0)
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, candidate in enumerate(confirmed.to_numpy(dtype=int)):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                # Opposite two-bar entry confirmation closes the position.
                position = 0
            positions[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = positions
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")

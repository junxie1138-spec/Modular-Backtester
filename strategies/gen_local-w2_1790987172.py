from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    path_window: int = 12
    participation_window: int = 60
    area_threshold: float = 0.06
    regime_band: float = 0.10
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1790987172"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        return int(max(max(2, params.path_window) + 1,
                       max(5, params.participation_window), 5))

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        window = max(2, int(params.path_window))
        baseline_window = max(5, int(params.participation_window))
        close = data["close"].astype(float).where(data["close"] > 0)
        volume = data["volume"].astype(float).where(data["volume"] > 0)
        x = np.log(close)
        y = np.log(volume)
        dx = x.diff()
        dy = y.diff()
        cross = x.shift(1) * dy - y.shift(1) * dx
        sum_dx = dx.rolling(window, min_periods=window).sum()
        sum_dy = dy.rolling(window, min_periods=window).sum()
        # Translate the path to its own window origin before computing area.
        area = 0.5 * (cross.rolling(window, min_periods=window).sum()
                      - x.shift(window) * sum_dy
                      + y.shift(window) * sum_dx)
        variation = (dx.abs().rolling(window, min_periods=window).sum()
                     * dy.abs().rolling(window, min_periods=window).sum())
        normalized_area = (area / variation.where(variation > 1e-12)).clip(-1.0, 1.0)
        baseline = volume.rolling(baseline_window, min_periods=baseline_window).mean()
        recent = volume.rolling(5, min_periods=5).mean()
        participation = recent / baseline.where(baseline > 0)
        valid = (normalized_area.notna() & sum_dx.notna() & participation.notna())
        out = pd.DataFrame(index=data.index)
        out["log_price"] = x
        out["log_volume"] = y
        out["price_increment"] = dx
        out["volume_increment"] = dy
        out["cross_increment"] = cross
        out["price_displacement"] = sum_dx
        out["volume_displacement"] = sum_dy
        out["path_variation"] = variation
        out["signature_area"] = area
        out["normalized_area"] = normalized_area
        out["volume_baseline"] = baseline
        out["recent_volume"] = recent
        out["participation"] = participation
        out["valid"] = valid.astype(int)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        threshold = max(float(params.area_threshold), 1e-6)
        band = min(max(float(params.regime_band), 0.0), 0.9)
        hold = max(1, int(params.holding_bars))
        area = indicators["normalized_area"]
        displacement = indicators["price_displacement"]
        direction = np.sign(displacement)
        ordering = area * direction
        participation = indicators["participation"]
        active = participation >= 1.0 + band
        quiet = participation <= 1.0 - band
        valid = ((indicators["valid"] > 0)
                 & (area.abs() >= threshold)
                 & (displacement.abs() > 1e-12))
        continuation = valid & active & (ordering < 0)
        exhaustion = valid & quiet & (ordering > 0)
        candidate = pd.Series(0, index=data.index, dtype=int)
        candidate.loc[continuation] = direction.loc[continuation].astype(int)
        candidate.loc[exhaustion] = -direction.loc[exhaustion].astype(int)
        conviction = (area.abs() / (3.0 * threshold)).clip(0.0, 1.0)
        entry_size = (0.25 + 0.70 * conviction).to_numpy(dtype=float)
        entries = candidate.to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        position = 0
        expiry = -1
        frozen_size = 0.25
        for i in range(len(data)):
            if position != 0 and i >= expiry:
                position = 0
                frozen_size = 0.25
                # Reserve the exit bar for a flat target; never reverse at expiry.
                continue
            if position == 0 and entries[i] != 0:
                position = int(entries[i])
                expiry = i + hold
                frozen_size = float(entry_size[i])
            signals[i] = position
            sizes[i] = frozen_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

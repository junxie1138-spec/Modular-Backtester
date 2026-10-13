from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AsymmetricAutocorrelationParams:
    window: int = 8
    entry_threshold: float = 0.30
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[AsymmetricAutocorrelationParams]):
    strategy_id = "gen_local_1790986015"

    @classmethod
    def params_type(cls) -> type[AsymmetricAutocorrelationParams]:
        return AsymmetricAutocorrelationParams

    def warmup_bars(self, params: AsymmetricAutocorrelationParams) -> int:
        window = max(3, min(8, int(params.window)))
        return window + 2

    def indicators(self, data: pd.DataFrame, params: AsymmetricAutocorrelationParams) -> pd.DataFrame:
        window = max(3, min(8, int(params.window)))
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lag_return"] = out["return"].shift(1)
        out["rms_return"] = out["return"].pow(2).rolling(window, min_periods=window).mean().pow(0.5)
        valid = out["return"].notna() & out["lag_return"].notna()
        for branch, positive in (("up", True), ("down", False)):
            mask = valid & (out["lag_return"].gt(0) if positive else out["lag_return"].lt(0))
            out[branch + "_pair_count"] = mask.astype(float).rolling(window, min_periods=window).sum()
            out[branch + "_cross_sum"] = (out["return"] * out["lag_return"]).where(mask, 0.0).rolling(window, min_periods=window).sum()
            out[branch + "_lag_energy"] = out["lag_return"].pow(2).where(mask, 0.0).rolling(window, min_periods=window).sum()
            out[branch + "_next_energy"] = out["return"].pow(2).where(mask, 0.0).rolling(window, min_periods=window).sum()
            denominator = (out[branch + "_lag_energy"] * out[branch + "_next_energy"]).pow(0.5).replace(0.0, np.nan)
            out[branch + "_autocorrelation"] = (out[branch + "_cross_sum"] / denominator).clip(-1.0, 1.0)
        out["selected_autocorrelation"] = out["up_autocorrelation"].where(out["return"].gt(0), out["down_autocorrelation"])
        out["selected_pair_count"] = out["up_pair_count"].where(out["return"].gt(0), out["down_pair_count"])
        out["forecast_score"] = (out["selected_autocorrelation"] * out["return"] / out["rms_return"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out["ready"] = valid & out["forecast_score"].notna() & out["selected_pair_count"].ge(2) & (np.arange(len(data)) >= self.warmup_bars(params) - 1)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AsymmetricAutocorrelationParams) -> SignalFrame:
        threshold = max(0.0, float(params.entry_threshold))
        holding_bars = max(1, int(params.holding_bars))
        scores = indicators["forecast_score"].fillna(0.0).to_numpy()
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                    continue
                signals[i] = position
                continue
            if ready[i] and abs(scores[i]) > threshold:
                position = 1 if scores[i] > 0.0 else -1
                exit_bar = i + holding_bars
                signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        # Both entry and scheduled exit move forward together by one bar.
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")

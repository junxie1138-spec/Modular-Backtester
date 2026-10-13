from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeBreathingParams:
    lookback: int = 8
    entry_threshold: float = 0.30
    reversal_threshold: float = 0.20
    target_volatility: float = 0.10


class GeneratedStrategy(BaseStrategy[RangeBreathingParams]):
    strategy_id = "gen_local_1790868161"

    @classmethod
    def params_type(cls) -> type[RangeBreathingParams]:
        return RangeBreathingParams

    def warmup_bars(self, params: RangeBreathingParams) -> int:
        return max(2, min(9, int(params.lookback))) + 1

    def indicators(self, data: pd.DataFrame, params: RangeBreathingParams) -> pd.DataFrame:
        window = max(2, min(9, int(params.lookback)))
        out = pd.DataFrame(index=data.index)
        out["center"] = (data["high"] + data["low"]) / 2.0
        out["width"] = (data["high"] - data["low"]).clip(lower=0.0)
        out["center_change"] = out["center"].diff()
        out["width_change"] = out["width"].diff()
        out["translation"] = out["center_change"].rolling(window, min_periods=window).sum()
        out["center_churn"] = out["center_change"].abs().rolling(window, min_periods=window).sum()
        out["breathing"] = out["width_change"].abs().rolling(window, min_periods=window).sum()
        out["noise"] = out["center_churn"] + 0.5 * out["breathing"]
        score = out["translation"].div(out["noise"].replace(0.0, np.nan))
        out["ready"] = score.notna() & np.isfinite(score)
        out["score"] = score.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1.0, 1.0)
        ratio = data["high"].div(data["low"].where(data["low"] > 0.0))
        out["log_range"] = np.log(ratio.where(ratio >= 1.0))
        out["range_variance"] = out["log_range"].pow(2).rolling(window, min_periods=window).mean()
        volatility = np.sqrt(out["range_variance"] * 252.0 / (4.0 * np.log(2.0)))
        out["volatility"] = volatility.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        target = max(0.001, float(params.target_volatility))
        out["size"] = (target / out["volatility"].clip(lower=0.001)).clip(lower=0.01, upper=0.95)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeBreathingParams,
    ) -> SignalFrame:
        entry = min(1.0, max(0.001, float(params.entry_threshold)))
        reversal = min(1.0, max(0.001, float(params.reversal_threshold)))
        signals = np.zeros(len(data), dtype=int)
        bullish = False
        scores = indicators["score"].to_numpy()
        ready = indicators["ready"].to_numpy(dtype=bool)
        for i in range(len(data)):
            if ready[i]:
                if scores[i] >= entry:
                    bullish = True
                elif scores[i] <= -reversal:
                    bullish = False
            signals[i] = int(bullish)
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = indicators["size"].shift(1).fillna(0.95).astype(float)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    lookback: int = 40
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local_1790995737"

    @classmethod
    def params_type(cls) -> type[MomentumParams]:
        return MomentumParams

    def warmup_bars(self, params: MomentumParams) -> int:
        return max(2, int(params.lookback)) + 1

    def indicators(self, data: pd.DataFrame, params: MomentumParams) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        returns = out["return"]
        out["return_sum"] = returns.rolling(window, min_periods=window).sum()
        out["return_energy"] = returns.pow(2).rolling(window, min_periods=window).sum()
        out["largest_return"] = returns.rolling(window, min_periods=window).max()
        out["smallest_return"] = returns.rolling(window, min_periods=window).min()
        out["long_residual"] = out["return_sum"] - out["largest_return"]
        out["short_residual"] = out["return_sum"] - out["smallest_return"]
        out["long_noise"] = (out["return_energy"] - out["largest_return"].pow(2)).clip(lower=0.0).pow(0.5)
        out["short_noise"] = (out["return_energy"] - out["smallest_return"].pow(2)).clip(lower=0.0).pow(0.5)
        out["long_score"] = out["long_residual"].div(out["long_noise"].where(out["long_noise"] > 0.0))
        out["short_score"] = out["short_residual"].div(out["short_noise"].where(out["short_noise"] > 0.0))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: MomentumParams) -> SignalFrame:
        long_entry = ((indicators["long_score"] >= 1.5) & (indicators["return_sum"] > 0.0)).fillna(False)
        short_entry = ((indicators["short_score"] <= -1.5) & (indicators["return_sum"] < 0.0)).fillna(False)
        candidates = np.select([long_entry.to_numpy(), short_entry.to_numpy()], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        holding = max(1, int(params.holding_bars))
        side = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if side != 0 and i >= exit_bar:
                side = 0
                continue
            if side == 0 and candidate != 0:
                side = int(candidate)
                exit_bar = i + holding
            signals[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["size"] = 0.95
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

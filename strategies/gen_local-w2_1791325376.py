from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SeasonalParams:
    history_bars: int = 504
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[SeasonalParams]):
    strategy_id = "gen_local-w2_1791325376"

    @classmethod
    def params_type(cls) -> type[SeasonalParams]:
        return SeasonalParams

    @staticmethod
    def warmup_bars(params: SeasonalParams) -> int:
        return int(params.history_bars + params.holding_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SeasonalParams) -> pd.DataFrame:
        if params.history_bars < 120 or not 3 <= params.holding_bars <= 5:
            raise ValueError("history_bars must be >= 120 and holding_bars must be 3-5")
        out = pd.DataFrame(index=data.index)
        out["calendar_bucket"] = data.index.dayofweek + 5 * (data.index.day > 15).astype(int)
        out["completed_entry_bucket"] = out["calendar_bucket"].shift(params.holding_bars)
        out["completed_return"] = data["close"].pct_change(params.holding_bars, fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["forecast"] = np.nan
        out["dispersion"] = np.nan
        for bucket in range(10):
            samples = out["completed_return"].where(out["completed_entry_bucket"].eq(bucket))
            out[f"bucket_{bucket}_mean"] = samples.rolling(params.history_bars, min_periods=12).mean()
            out[f"bucket_{bucket}_std"] = samples.rolling(params.history_bars, min_periods=12).std()
            selected = out["calendar_bucket"].eq(bucket)
            out.loc[selected, "forecast"] = out.loc[selected, f"bucket_{bucket}_mean"]
            out.loc[selected, "dispersion"] = out.loc[selected, f"bucket_{bucket}_std"]
        out["seasonal_score"] = (out["forecast"] / out["dispersion"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        out.iloc[:GeneratedStrategy.warmup_bars(params), out.columns.get_loc("seasonal_score")] = np.nan
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SeasonalParams) -> SignalFrame:
        score = indicators["seasonal_score"].fillna(0.0).to_numpy()
        candidates = np.where(score > 0.15, 1, np.where(score < -0.15, -1, 0))
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i - entry_bar >= params.holding_bars:
                    position = 0
            elif candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
            signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["size"] = 1.0
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")

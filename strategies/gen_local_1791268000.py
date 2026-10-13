from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class IsolationParams:
    percentile_window: int = 126
    downside_quantile: float = 0.08
    quiet_bars: int = 4
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[IsolationParams]):
    strategy_id = "gen_local_1791268000"

    @classmethod
    def params_type(cls) -> type[IsolationParams]:
        return IsolationParams

    def warmup_bars(self, params: IsolationParams) -> int:
        return max(params.percentile_window, params.quiet_bars) + 2

    def indicators(self, data: pd.DataFrame, params: IsolationParams) -> pd.DataFrame:
        if params.percentile_window < 2 or params.quiet_bars < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not 0.0 < params.downside_quantile < 0.5:
            raise ValueError("downside_quantile must lie between zero and 0.5.")
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five.")
        close = data["close"].where(data["close"] > 0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        previous = returns.shift(1)
        history = previous.rolling(params.percentile_window, min_periods=params.percentile_window)
        quiet = previous.rolling(params.quiet_bars, min_periods=params.quiet_bars)
        positive = previous.gt(0).astype(float).where(previous.notna())
        out = pd.DataFrame(index=data.index)
        out["return"] = returns
        out["downside_threshold"] = history.quantile(params.downside_quantile)
        out["quiet_threshold"] = previous.abs().rolling(params.percentile_window, min_periods=params.percentile_window).quantile(0.5)
        out["prior_max_abs_return"] = previous.abs().rolling(params.quiet_bars, min_periods=params.quiet_bars).max()
        out["prior_min_return"] = quiet.min()
        out["prior_positive_count"] = positive.rolling(params.quiet_bars, min_periods=params.quiet_bars).sum()
        out["entry"] = (
            out["return"].lt(0)
            & out["return"].lt(out["downside_threshold"])
            & out["prior_max_abs_return"].le(out["quiet_threshold"])
            & out["prior_min_return"].lt(0)
            & out["prior_positive_count"].gt(0)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: IsolationParams) -> SignalFrame:
        if not 3 <= params.holding_bars <= 5:
            raise ValueError("holding_bars must be between three and five.")
        entries = indicators["entry"].reindex(data.index).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i < exit_bar:
                    signal[i] = 1
                else:
                    exit_bar = -1
                continue
            if enter:
                signal[i] = 1
                exit_bar = i + params.holding_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    entries = pd.DataFrame({"entry": [True, True, False, False, True, False, True, False, False, False, False, False]}, index=index)
    result = GeneratedStrategy().generate_signals(data, entries, None, IsolationParams()).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
    assert result["size"].gt(0).all()

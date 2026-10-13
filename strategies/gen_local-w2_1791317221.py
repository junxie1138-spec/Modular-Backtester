from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class QuietCarrierParams:
    window: int = 20
    efficiency_threshold: float = 0.35
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[QuietCarrierParams]):
    strategy_id = "gen_local-w2_1791317221"

    @classmethod
    def params_type(cls) -> type[QuietCarrierParams]:
        return QuietCarrierParams

    def warmup_bars(self, params: QuietCarrierParams) -> int:
        return 2 * params.window + 2

    def indicators(self, data: pd.DataFrame, params: QuietCarrierParams) -> pd.DataFrame:
        if params.window < 2:
            raise ValueError("window must be at least 2")
        if not 0.0 < params.efficiency_threshold < 1.0:
            raise ValueError("efficiency_threshold must be between 0 and 1")
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        w = params.window
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["absolute_return"] = out["return"].abs()
        # Classify today's magnitude using only preceding observations.
        out["quiet_ceiling"] = out["absolute_return"].rolling(w, min_periods=w).median().shift(1)
        ready = out["quiet_ceiling"].notna() & out["return"].notna()
        quiet = ready & out["absolute_return"].le(out["quiet_ceiling"])
        out["quiet_flag"] = quiet.astype(float).where(ready)
        out["quiet_return"] = out["return"].where(quiet, 0.0).where(ready)
        out["quiet_absolute_return"] = out["quiet_return"].abs()
        out["total_drift"] = out["return"].rolling(w, min_periods=w).sum()
        out["quiet_drift"] = out["quiet_return"].rolling(w, min_periods=w).sum()
        out["quiet_gross"] = out["quiet_absolute_return"].rolling(w, min_periods=w).sum()
        out["quiet_count"] = out["quiet_flag"].rolling(w, min_periods=w).sum()
        out["quiet_efficiency"] = out["quiet_drift"].div(out["quiet_gross"].where(out["quiet_gross"] > 0.0))
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: QuietCarrierParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive")
        supported = indicators["quiet_count"].ge(max(2, int(np.ceil(params.window * 0.25))))
        long_entry = supported & indicators["total_drift"].gt(0.0) & indicators["quiet_efficiency"].ge(params.efficiency_threshold)
        short_entry = supported & indicators["total_drift"].lt(0.0) & indicators["quiet_efficiency"].le(-params.efficiency_threshold)
        candidate = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = candidate.where(candidate.eq(candidate.shift(1)), 0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i, direction in enumerate(confirmed):
            if position != 0:
                if i - entry_bar == params.holding_bars:
                    position = 0
                    # Preserve an explicit flat target on the exit bar.
                    continue
            elif direction != 0:
                position = int(direction)
                entry_bar = i
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = QuietCarrierParams()
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    directions = np.array([1, 1, -1, -1, -1, -1, -1, -1])
    features = pd.DataFrame({"total_drift": directions * 0.01, "quiet_efficiency": directions * 0.8, "quiet_count": 10.0}, index=index)
    # This method uses no instance state; exercise its timing directly.
    result = GeneratedStrategy.generate_signals(None, data, features, None, params)
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 0, -1]
    assert result.data["size"].gt(0.0).all()

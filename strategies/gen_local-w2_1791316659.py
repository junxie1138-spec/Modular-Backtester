from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    ma_bars: int = 21
    correlation_bars: int = 60
    entry_z: float = 1.0
    correlation_ceiling: float = -0.10
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791316659"

    @classmethod
    def params_type(cls) -> type[Params]:
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(2 * params.ma_bars + 1, params.correlation_bars + 3))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.ma_bars < 2 or params.correlation_bars < 3 or params.hold_bars < 1:
            raise ValueError("Lookbacks and holding period must be positive and sufficiently long")
        if not np.isfinite(params.entry_z) or params.entry_z <= 0:
            raise ValueError("entry_z must be finite and positive")
        if not np.isfinite(params.correlation_ceiling) or not -1 < params.correlation_ceiling < 0:
            raise ValueError("correlation_ceiling must be between -1 and 0")
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        out["ma"] = close.rolling(params.ma_bars, min_periods=params.ma_bars).mean()
        out["distance"] = close.div(out["ma"].replace(0, np.nan)).sub(1)
        out["distance_scale"] = out["distance"].rolling(params.ma_bars, min_periods=params.ma_bars).std(ddof=0)
        out["z"] = out["distance"].div(out["distance_scale"].replace(0, np.nan))
        out["return"] = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["lagged_return"] = out["return"].shift(1)
        out["autocorrelation"] = out["return"].rolling(params.correlation_bars, min_periods=params.correlation_bars).corr(out["lagged_return"]).replace([np.inf, -np.inf], np.nan)
        out["z_change"] = out["z"].diff()
        out["candidate"] = (out["z"].lt(-params.entry_z) & out["z_change"].gt(0) & out["autocorrelation"].le(params.correlation_ceiling))
        out["confirmed"] = out["candidate"] & out["candidate"].shift(1, fill_value=False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        entries = indicators["confirmed"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        remaining = 0
        for i, entry in enumerate(entries):
            if remaining > 0:
                remaining -= 1
                signals[i] = int(remaining > 0)
            elif entry:
                remaining = params.hold_bars
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = Params()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(index=index)
    features = pd.DataFrame({"confirmed": [False, True] + [False] * 10}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, features, None, params)
    assert result.data["signal"].tolist() == [0, 0] + [1] * 8 + [0, 0]

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShiftParams:
    ma_window: int = 60
    block_window: int = 20
    z_shift_threshold: float = 0.35
    body_shift_threshold: float = 0.05


class GeneratedStrategy(BaseStrategy[ShiftParams]):
    strategy_id = "gen_local_1790872387"

    @classmethod
    def params_type(cls) -> type[ShiftParams]:
        return ShiftParams

    def warmup_bars(self, params: ShiftParams) -> int:
        return int(params.ma_window + 2 * params.block_window)

    def indicators(self, data: pd.DataFrame, params: ShiftParams) -> pd.DataFrame:
        if params.ma_window < 2 or params.block_window < 1:
            raise ValueError("ma_window must be >= 2 and block_window >= 1")
        if not np.isfinite(params.z_shift_threshold) or not np.isfinite(params.body_shift_threshold):
            raise ValueError("Thresholds must be finite")
        if params.z_shift_threshold < 0 or params.body_shift_threshold < 0:
            raise ValueError("Thresholds must be nonnegative")
        out = pd.DataFrame(index=data.index)
        out["ma"] = data["close"].rolling(params.ma_window, min_periods=params.ma_window).mean()
        out["price_std"] = data["close"].rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        out["z"] = (data["close"] - out["ma"]) / out["price_std"].where(out["price_std"] > 0)
        out["range"] = data["high"] - data["low"]
        out["body"] = (data["close"] - data["open"]) / out["range"].where(out["range"] > 0)
        out["body"] = out["body"].mask(out["range"].eq(0) & data["close"].eq(data["open"]), 0.0)
        out = out.replace([np.inf, -np.inf], np.nan)
        for primitive in ("z", "body"):
            out[primitive + "_recent"] = out[primitive].rolling(params.block_window, min_periods=params.block_window).mean()
            out[primitive + "_previous"] = out[primitive + "_recent"].shift(params.block_window)
            out[primitive + "_shift"] = out[primitive + "_recent"] - out[primitive + "_previous"]
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: ShiftParams) -> SignalFrame:
        valid = np.isfinite(indicators["z_shift"]) & np.isfinite(indicators["body_shift"])
        entry = valid & (indicators["z_shift"] > params.z_shift_threshold) & (indicators["body_shift"] > params.body_shift_threshold)
        df = pd.DataFrame(index=data.index)
        # Remain long exactly while both entry primitives agree; their conjunction reversing exits.
        df["signal"] = entry.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = ShiftParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=6, freq="B")
    data = pd.DataFrame(100.0, index=index, columns=["open", "high", "low", "close", "volume"])
    ind = pd.DataFrame({"z_shift": [np.nan, 0.6, 0.6, 0.1, 0.6, 0.6], "body_shift": [np.nan, 0.0, 0.2, 0.2, 0.2, -0.2]}, index=index)
    result = strategy.generate_signals(data, ind, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 1]
    assert result["size"].gt(0).all()
    index = pd.date_range("2020-01-01", periods=180, freq="B")
    close = 100.0 + np.arange(180) * 0.1 + np.sin(np.arange(180))
    data = pd.DataFrame({"open": close - 0.1, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000000.0}, index=index)
    ind = strategy.indicators(data, params)
    assert ind["z_shift"].first_valid_index() == index[98]
    assert strategy.warmup_bars(params) >= 99
    changed = data.copy()
    changed.loc[index[130]:, "close"] += 20.0
    pd.testing.assert_frame_equal(ind.iloc[:130], strategy.indicators(changed, params).iloc[:130])

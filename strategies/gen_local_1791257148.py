from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    depth_center: float = 1.0
    reversal_band: float = 0.25
    erosion_threshold: float = 0.35

    def __post_init__(self):
        values = (self.depth_center, self.reversal_band, self.erosion_threshold)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Parameters must be finite and positive.")


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791257148"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 10

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["peak"] = data["high"].rolling(8, min_periods=8).max()
        out["range_scale"] = (data["high"] - data["low"]).rolling(4, min_periods=4).mean().replace(0.0, np.nan)
        out["drawdown_depth"] = (out["peak"] - data["close"]).clip(lower=0.0) / out["range_scale"]
        out["peak_erosion"] = -out["peak"].diff(2) / out["range_scale"]
        out["ready"] = np.isfinite(out[["drawdown_depth", "peak_erosion"]]).all(axis=1)
        out["eroding_regime"] = out["peak_erosion"] > params.erosion_threshold
        out["force"] = (out["drawdown_depth"] - params.depth_center) * np.where(out["eroding_regime"], -1.0, 1.0)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        force = indicators["force"]
        candidates = np.where(force > params.reversal_band, 1, np.where(force < -params.reversal_band, -1, 0))
        candidates = np.where(indicators["ready"], candidates, 0)
        position = 0
        signals = np.zeros(len(data), dtype=int)
        for i, candidate in enumerate(candidates):
            if position == 0:
                position = int(candidate)
            elif candidate == -position:
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=6)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"force": [1.0, 0.0, -1.0, -1.0, 1.0, 0.0], "ready": True}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=10)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, 0]
    assert (result["size"] > 0).all()
    assert strategy.warmup_bars(params) == 10


if __name__ == "__main__":
    _self_check()

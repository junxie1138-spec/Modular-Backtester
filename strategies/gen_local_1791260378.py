from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_window: int = 5
    smoothing: int = 3
    threshold: float = 0.15


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1791260378"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    @staticmethod
    def warmup_bars(params: AccelerationParams) -> int:
        return int(max(200, 2 * params.roc_window + params.smoothing))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if params.roc_window < 1 or params.smoothing < 1 or not np.isfinite(params.threshold) or params.threshold <= 0:
            raise ValueError("Windows must be positive and threshold must be finite and positive")
        close = data["close"].astype(float).where(lambda s: s > 0)
        daily_return = close.pct_change(fill_method=None)
        roc = close.pct_change(params.roc_window, fill_method=None)
        previous_roc = roc.shift(params.roc_window)
        acceleration = roc - previous_roc
        path_activity = daily_return.abs().rolling(2 * params.roc_window, min_periods=2 * params.roc_window).sum()
        normalized_acceleration = acceleration.div(path_activity.where(path_activity > 0))
        score = normalized_acceleration.rolling(params.smoothing, min_periods=params.smoothing).mean()
        ma200 = close.rolling(200, min_periods=200).mean()
        return pd.DataFrame({
            "daily_return": daily_return,
            "roc": roc,
            "previous_roc": previous_roc,
            "acceleration": acceleration,
            "path_activity": path_activity,
            "normalized_acceleration": normalized_acceleration,
            "score": score,
            "ma200": ma200,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        score = indicators["score"]
        valid = score.notna() & indicators["ma200"].notna() & data["close"].notna()
        direction = np.where(valid & (score > params.threshold), 1,
                             np.where(valid & (score < -params.threshold), -1, 0))
        long_regime = (data["close"] > indicators["ma200"]).to_numpy()
        short_regime = (data["close"] < indicators["ma200"]).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, pulse in enumerate(direction):
            if position == 0:
                if pulse == 1 and long_regime[i]:
                    position = 1
                elif pulse == -1 and short_regime[i]:
                    position = -1
            elif pulse == -position:
                # Opposite entry primitive exits to flat; the MA gates new entries only.
                position = 0
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [101.0, 101.0, 101.0, 99.0, 99.0, 99.0, 101.0]}, index=index)
    indicators = pd.DataFrame({"score": [0.3, np.nan, -0.3, -0.3, 0.0, 0.3, 0.3], "ma200": 100.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, AccelerationParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert (result.data["size"] > 0).all()
    assert GeneratedStrategy.warmup_bars(AccelerationParams()) >= 200


if __name__ == "__main__":
    _self_check()

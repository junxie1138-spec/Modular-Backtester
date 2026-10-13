from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AcceptanceParams:
    lookback: int = 20
    breakout_z: float = 1.5


class GeneratedStrategy(BaseStrategy[AcceptanceParams]):
    strategy_id = "gen_local_1790890221"

    @classmethod
    def params_type(cls) -> type[AcceptanceParams]:
        return AcceptanceParams

    @staticmethod
    def warmup_bars(params: AcceptanceParams) -> int:
        return max(2, int(params.lookback)) + 1

    @staticmethod
    def indicators(data: pd.DataFrame, params: AcceptanceParams) -> pd.DataFrame:
        window = max(2, int(params.lookback))
        close = data["close"].astype(float)
        mean = close.rolling(window, min_periods=window).mean().shift(1)
        deviation = close.rolling(window, min_periods=window).std(ddof=0).shift(1)
        deviation = deviation.where(deviation > 0.0)
        result = pd.DataFrame(index=data.index)
        result["reference_mean"] = mean
        result["reference_std"] = deviation
        for column in ("low", "high", "close"):
            result["z_" + column] = ((data[column] - mean) / deviation).replace([np.inf, -np.inf], np.nan)
        return result

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: AcceptanceParams,
    ) -> SignalFrame:
        threshold = float(params.breakout_z)
        if not np.isfinite(threshold) or threshold <= 0.5:
            raise ValueError("breakout_z must be finite and greater than 0.5")
        z_close = indicators["z_close"].to_numpy(dtype=float)
        z_low = indicators["z_low"].to_numpy(dtype=float)
        z_high = indicators["z_high"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        armed = False
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar == 7:
                    position = 0
                signals[bar] = position
                continue
            if not (np.isfinite(z_close[bar]) and np.isfinite(z_low[bar]) and np.isfinite(z_high[bar])):
                continue
            if abs(z_close[bar]) <= 0.5:
                armed = True
            if armed:
                if z_low[bar] > threshold:
                    position = 1
                elif z_high[bar] < -threshold:
                    position = -1
                if position != 0:
                    entry_bar = bar
                    armed = False
            signals[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    data = pd.DataFrame(index=index)
    z = np.array([0, 2, 2, -2, -2, -2, -2, -2, -2, 0, -2, -2], dtype=float)
    indicators = pd.DataFrame({"z_close": z, "z_low": z - 0.1, "z_high": z + 0.1}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, AcceptanceParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, -1]
    assert (result.data["size"] > 0).all()

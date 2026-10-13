from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CalendarAccelerationParams:
    roc_bars: int = 5
    dispersion_bars: int = 40
    entry_threshold: float = 0.35


class GeneratedStrategy(BaseStrategy[CalendarAccelerationParams]):
    strategy_id = "gen_local_1791339471"

    @classmethod
    def params_type(cls) -> type[CalendarAccelerationParams]:
        return CalendarAccelerationParams

    def warmup_bars(self, params: CalendarAccelerationParams) -> int:
        return int(2 * max(1, params.roc_bars) + 3 + max(2, params.dispersion_bars) + 1)

    def indicators(self, data: pd.DataFrame, params: CalendarAccelerationParams) -> pd.DataFrame:
        lag = max(1, int(params.roc_bars))
        window = max(2, int(params.dispersion_bars))
        close = data["close"].where(data["close"] > 0)
        roc = close.pct_change(lag, fill_method=None)
        acceleration = roc.diff(lag)
        smooth_acceleration = acceleration.rolling(3, min_periods=3).mean()
        dispersion = smooth_acceleration.rolling(window, min_periods=window).std(ddof=0)
        phase = pd.Series(
            (data.index.day - 1) / data.index.days_in_month,
            index=data.index,
            dtype=float,
        )
        seasonal_polarity = np.cos(2.0 * np.pi * phase)
        score = seasonal_polarity * smooth_acceleration / dispersion.where(dispersion > 0)
        score = score.replace([np.inf, -np.inf], np.nan)
        ready = score.notna()
        return pd.DataFrame({
            "roc": roc,
            "acceleration": acceleration,
            "smooth_acceleration": smooth_acceleration,
            "dispersion": dispersion,
            "calendar_phase": phase,
            "seasonal_polarity": seasonal_polarity,
            "score": score,
            "ready": ready,
        }, index=data.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: CalendarAccelerationParams,
    ) -> SignalFrame:
        threshold = max(0.0, float(params.entry_threshold))
        score = indicators["score"].fillna(0.0)
        ready = indicators["ready"].fillna(False).astype(bool)
        direction = pd.Series(0, index=data.index, dtype=int)
        direction.loc[ready & (score > threshold)] = 1
        direction.loc[ready & (score < -threshold)] = -1
        confirmed = direction.ne(0) & direction.eq(direction.shift(1))
        positions = np.zeros(len(data), dtype=int)
        position = 0
        directions = direction.to_numpy()
        confirmations = confirmed.to_numpy()
        for i in range(len(data)):
            candidate = int(directions[i])
            if position != 0:
                if candidate == -position:
                    position = 0
            elif confirmations[i]:
                position = candidate
            positions[i] = position
        df = pd.DataFrame({"signal": positions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"score": [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0], "ready": True}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, CalendarAccelerationParams())
    assert result.data["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1]

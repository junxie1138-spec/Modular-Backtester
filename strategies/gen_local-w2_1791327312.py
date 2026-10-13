from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeBreathingParams:
    vol_window: int = 12
    pressure_threshold: float = 0.08
    target_daily_vol: float = 0.008
    min_size: float = 0.05
    max_size: float = 0.95


class GeneratedStrategy(BaseStrategy[RangeBreathingParams]):
    strategy_id = "gen_local-w2_1791327312"

    @classmethod
    def params_type(cls) -> type[RangeBreathingParams]:
        return RangeBreathingParams

    @staticmethod
    def warmup_bars(params: RangeBreathingParams) -> int:
        if not isinstance(params.vol_window, int) or params.vol_window < 2:
            raise ValueError("vol_window must be an integer >= 2")
        values = [params.pressure_threshold, params.target_daily_vol, params.min_size, params.max_size]
        if not np.isfinite(values).all():
            raise ValueError("Parameters must be finite")
        if not 0 < params.pressure_threshold < 1:
            raise ValueError("pressure_threshold must be between zero and one")
        if params.target_daily_vol <= 0 or not 0 < params.min_size <= params.max_size <= 1:
            raise ValueError("Invalid volatility target or exposure bounds")
        return int(max(2, params.vol_window))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeBreathingParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        high = data["high"].astype(float)
        low = data["low"].astype(float)
        close = data["close"].astype(float)
        valid = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
        valid &= (low > 0) & (high > low) & (close >= low) & (close <= high)
        width = (high - low).where(valid)
        relative_range = (2.0 * width / (high + low)).where(valid)
        previous_range = relative_range.shift(1)
        breathing = (relative_range - previous_range) / (relative_range + previous_range)
        closing_pressure = ((2.0 * close - high - low) / width).clip(-1.0, 1.0)
        score = (breathing * closing_pressure).replace([np.inf, -np.inf], np.nan)
        log_range_squared = np.log(high.where(valid) / low.where(valid)).pow(2)
        daily_vol = np.sqrt(log_range_squared.rolling(params.vol_window, min_periods=params.vol_window).mean() / (4.0 * np.log(2.0)))
        ready = score.notna() & daily_vol.notna() & (daily_vol > 0)
        exposure = (params.target_daily_vol / daily_vol.where(daily_vol > 0)).replace([np.inf, -np.inf], np.nan)
        exposure = exposure.clip(params.min_size, params.max_size).fillna(params.min_size)
        vote = pd.Series(0, index=data.index, dtype=int)
        vote.loc[ready & (score >= params.pressure_threshold)] = 1
        vote.loc[ready & (score <= -params.pressure_threshold)] = -1
        return pd.DataFrame({
            "relative_range": relative_range.fillna(0.0),
            "previous_range": previous_range.fillna(0.0),
            "breathing": breathing.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "closing_pressure": closing_pressure.replace([np.inf, -np.inf], np.nan).fillna(0.0),
            "score": score.fillna(0.0),
            "log_range_squared": log_range_squared.fillna(0.0),
            "daily_vol": daily_vol.fillna(0.0),
            "ready": ready.astype(int),
            "vote": vote,
            "exposure": exposure,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeBreathingParams) -> SignalFrame:
        votes = indicators["vote"].fillna(0).astype(int).to_numpy()
        positions = np.zeros(len(data), dtype=int)
        position = 0
        for i, vote in enumerate(votes):
            if position == 0:
                position = int(vote)
            elif vote == -position:
                # Opposite entry qualification exits to flat; neutral votes retain the position.
                position = 0
            positions[i] = position
        df = pd.DataFrame({"signal": positions}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["exposure"].shift(1).fillna(params.min_size).clip(params.min_size, params.max_size).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = RangeBreathingParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"vote": [1, 0, -1, -1, 0, 1, 1], "exposure": np.arange(1, 8) / 10.0}, index=index)
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"], [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    indicators["vote"] *= -1
    mirrored = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert mirrored["signal"].equals(-result["signal"])


if __name__ == "__main__":
    _self_check()

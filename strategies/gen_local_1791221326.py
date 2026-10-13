from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AccelerationParams:
    roc_bars: int = 10
    vol_bars: int = 20
    coherence_threshold: float = 0.5
    acceleration_threshold: float = 0.5
    target_vol: float = 0.12


class GeneratedStrategy(BaseStrategy[AccelerationParams]):
    strategy_id = "gen_local_1791221326"

    @classmethod
    def params_type(cls) -> type[AccelerationParams]:
        return AccelerationParams

    @staticmethod
    def warmup_bars(params: AccelerationParams) -> int:
        return int(max(2 * params.roc_bars + 1, params.vol_bars + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: AccelerationParams) -> pd.DataFrame:
        if params.roc_bars < 1 or params.vol_bars < 2:
            raise ValueError("Lookbacks must be positive and vol_bars must be at least two")
        if not 0.0 < params.coherence_threshold <= 1.0:
            raise ValueError("coherence_threshold must be in (0, 1]")
        if not np.isfinite(params.acceleration_threshold) or params.acceleration_threshold <= 0.0:
            raise ValueError("acceleration_threshold must be finite and positive")
        if not np.isfinite(params.target_vol) or params.target_vol <= 0.0:
            raise ValueError("target_vol must be finite and positive")
        close = data["close"].where(data["close"] > 0.0)
        returns = close.pct_change(fill_method=None)
        roc = close.pct_change(params.roc_bars, fill_method=None)
        roc_step = roc.diff()
        acceleration = roc.diff(params.roc_bars)
        path = roc_step.abs().rolling(params.roc_bars, min_periods=params.roc_bars).sum()
        daily_vol = returns.rolling(params.vol_bars, min_periods=params.vol_bars).std(ddof=0)
        out = pd.DataFrame(index=data.index)
        out["returns"] = returns
        out["roc"] = roc
        out["roc_step"] = roc_step
        out["acceleration"] = acceleration
        out["roc_path"] = path
        out["daily_vol"] = daily_vol
        out["coherence"] = acceleration.div(path.where(path > 0.0)).clip(-1.0, 1.0)
        out["acceleration_score"] = acceleration.div(daily_vol.where(daily_vol > 0.0) * np.sqrt(2.0 * params.roc_bars))
        out["annual_vol"] = daily_vol * np.sqrt(252.0)
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: AccelerationParams) -> SignalFrame:
        coherence = indicators["coherence"]
        score = indicators["acceleration_score"]
        annual_vol = indicators["annual_vol"]
        ready = np.isfinite(coherence) & np.isfinite(score) & np.isfinite(annual_vol) & annual_vol.gt(0.0)
        long_entry = ready & coherence.ge(params.coherence_threshold) & score.ge(params.acceleration_threshold)
        short_entry = ready & coherence.le(-params.coherence_threshold) & score.le(-params.acceleration_threshold)
        direction = pd.Series(np.select([long_entry, short_entry], [1, -1], default=0), index=data.index, dtype=int)
        confirmed = direction.ne(0) & direction.eq(direction.shift(1))
        signals = np.zeros(len(data), dtype=int)
        position = 0
        for i, (candidate, entry_confirmed) in enumerate(zip(direction.to_numpy(), confirmed.to_numpy())):
            if position != 0:
                if candidate == -position:
                    position = 0
            elif entry_confirmed:
                position = int(candidate)
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        exposure = params.target_vol / annual_vol.where(annual_vol > 0.0)
        exposure = exposure.replace([np.inf, -np.inf], np.nan).clip(0.05, 0.95).fillna(0.05)
        df["size"] = exposure.shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = AccelerationParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    direction = np.array([1, 1, 0, -1, -1, -1, 1, 1])
    features = pd.DataFrame({"coherence": direction.astype(float), "acceleration_score": direction.astype(float), "annual_vol": 0.24}, index=index)
    result = GeneratedStrategy.generate_signals(data, features, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, -1, -1, 0]
    assert np.allclose(result["size"].iloc[1:], 0.5)
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=60))
    flat_features = GeneratedStrategy.indicators(flat, params)
    flat_result = GeneratedStrategy.generate_signals(flat, flat_features, None, params).data
    assert flat_result["signal"].eq(0).all()
    assert np.isfinite(flat_result["size"]).all() and flat_result["size"].gt(0.0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 2 * params.roc_bars + 1


if __name__ == "__main__":
    _self_check()

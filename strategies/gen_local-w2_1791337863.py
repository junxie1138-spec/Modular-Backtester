from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class TransportParams:
    scale_window: int = 12
    flux_threshold: float = 0.65


class GeneratedStrategy(BaseStrategy[TransportParams]):
    strategy_id = "gen_local-w2_1791337863"

    @classmethod
    def params_type(cls) -> type[TransportParams]:
        return TransportParams

    @staticmethod
    def warmup_bars(params: TransportParams) -> int:
        return max(2, params.scale_window) + 2

    @staticmethod
    def indicators(data: pd.DataFrame, params: TransportParams) -> pd.DataFrame:
        if params.scale_window < 2 or not np.isfinite(params.flux_threshold) or params.flux_threshold <= 0:
            raise ValueError("scale_window must be >= 2 and flux_threshold must be finite and positive")
        out = pd.DataFrame(index=data.index)
        out["return"] = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["previous_return"] = out["return"].shift(1)
        out["scale"] = out["return"].abs().rolling(params.scale_window, min_periods=params.scale_window).mean().shift(1)
        same_direction = out["return"] * out["previous_return"] > 0
        denominator = (out["return"] + out["previous_return"]).where(same_direction)
        out["transport"] = (2.0 * out["return"] * out["previous_return"] / denominator).where(same_direction, 0.0)
        valid = out["return"].notna() & out["previous_return"].notna() & out["scale"].gt(0)
        out["flux"] = (out["transport"] / out["scale"].where(out["scale"].gt(0))).where(valid).replace([np.inf, -np.inf], np.nan)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: TransportParams) -> SignalFrame:
        flux = indicators["flux"]
        entry = flux.ge(params.flux_threshold)
        exit_signal = flux.le(-params.flux_threshold)
        transitions = pd.Series(np.nan, index=data.index, dtype=float)
        transitions.loc[entry] = 1.0
        transitions.loc[exit_signal] = 0.0
        df = pd.DataFrame(index=data.index)
        df["signal"] = transitions.ffill().fillna(0).astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = TransportParams(scale_window=4, flux_threshold=0.65)
    returns = np.concatenate([np.full(12, -0.01), np.full(12, 0.01), np.full(12, -0.01)])
    close = 100.0 * np.cumprod(1.0 + returns)
    data = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=len(close)))
    indicators = GeneratedStrategy.indicators(data, params)
    signals = GeneratedStrategy.generate_signals(data, indicators, None, params).data["signal"]
    assert signals.iloc[13] == 0 and signals.iloc[14] == 1
    assert signals.iloc[25] == 1 and signals.iloc[26] == 0
    assert signals.isin([0, 1]).all()

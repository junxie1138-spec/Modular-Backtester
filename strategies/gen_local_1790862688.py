from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRegimeParams:
    regime_window: int = 40
    history_window: int = 126
    gap_quantile: float = 0.85
    regime_quantile: float = 0.70
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[GapRegimeParams]):
    strategy_id = "gen_local_1790862688"

    @classmethod
    def params_type(cls) -> type[GapRegimeParams]:
        return GapRegimeParams

    @staticmethod
    def warmup_bars(params: GapRegimeParams) -> int:
        if params.regime_window < 2 or params.history_window < 2:
            raise ValueError("Lookback windows must be at least two bars.")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive.")
        if not 0.5 < params.gap_quantile < 1.0:
            raise ValueError("gap_quantile must be between 0.5 and 1.")
        if not 0.5 < params.regime_quantile < 1.0:
            raise ValueError("regime_quantile must be between 0.5 and 1.")
        return int(params.regime_window + params.history_window + 2)

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRegimeParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1).where(lambda s: s > 0)
        out["gap"] = (data["open"] / out["prior_close"] - 1.0).replace([np.inf, -np.inf], np.nan)
        out["intraday"] = (data["close"] / data["open"].where(data["open"] > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["abs_gap"] = out["gap"].abs()
        out["gap_threshold"] = out["abs_gap"].rolling(params.history_window, min_periods=params.history_window).quantile(params.gap_quantile).shift(1)
        out["correlation"] = out["gap"].rolling(params.regime_window, min_periods=params.regime_window).corr(out["intraday"]).replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)
        out["regime"] = out["correlation"].shift(1)
        out["regime_upper"] = out["regime"].rolling(params.history_window, min_periods=params.history_window).quantile(params.regime_quantile).shift(1)
        out["regime_lower"] = out["regime"].rolling(params.history_window, min_periods=params.history_window).quantile(1.0 - params.regime_quantile).shift(1)
        out["gap_direction"] = np.sign(out["gap"].fillna(0.0)).astype(int)
        out["ready"] = out[["gap", "intraday", "gap_threshold", "regime", "regime_upper", "regime_lower"]].notna().all(axis=1)
        out["extreme"] = out["ready"] & (out["abs_gap"] > out["gap_threshold"])
        out["acceptance"] = (out["regime"] > 0.0) & (out["regime"] > out["regime_upper"])
        out["rejection"] = (out["regime"] < 0.0) & (out["regime"] < out["regime_lower"])
        out["candidate"] = np.select(
            [out["extreme"] & out["acceptance"], out["extreme"] & out["rejection"]],
            [out["gap_direction"], -out["gap_direction"]],
            default=0,
        ).astype(int)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: GapRegimeParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        candidates = indicators["candidate"].reindex(data.index).fillna(0).to_numpy(dtype=int)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, candidate in enumerate(candidates):
            if position != 0:
                if i == exit_bar:
                    position = 0
            elif candidate != 0:
                position = int(candidate)
                exit_bar = i + params.hold_bars
            signals[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapRegimeParams()
    index = pd.date_range("2020-01-01", periods=50, freq="B")
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    candidates = pd.DataFrame({"candidate": [1] + [-1] * 49}, index=index)
    result = GeneratedStrategy.generate_signals(data, candidates, None, params).data
    assert result["signal"].iloc[0] == 0
    assert (result["signal"].iloc[1:19] == 1).all()
    assert result["signal"].iloc[19] == 0
    assert (result["signal"].iloc[20:38] == -1).all()
    assert result["signal"].iloc[38] == 0
    assert (result["size"] > 0.0).all()
    flat_indicators = GeneratedStrategy.indicators(data, params)
    flat_result = GeneratedStrategy.generate_signals(data, flat_indicators, None, params).data
    assert (flat_result["signal"] == 0).all()

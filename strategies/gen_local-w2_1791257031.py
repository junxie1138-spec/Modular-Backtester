from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DisplacementParams:
    atr_window: int = 14
    trend_window: int = 12
    gap_atr_min: float = 0.3
    efficiency_min: float = 0.35
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[DisplacementParams]):
    strategy_id = "gen_local-w2_1791257031"

    @classmethod
    def params_type(cls):
        return DisplacementParams

    def warmup_bars(self, params: DisplacementParams) -> int:
        return int(max(params.atr_window + 2, params.trend_window + 1))

    def indicators(self, data: pd.DataFrame, params: DisplacementParams) -> pd.DataFrame:
        if params.atr_window < 2 or params.trend_window < 2:
            raise ValueError("Indicator windows must be at least two bars.")
        if not np.isfinite(params.gap_atr_min) or params.gap_atr_min <= 0:
            raise ValueError("gap_atr_min must be finite and positive.")
        if not np.isfinite(params.efficiency_min) or not 0 < params.efficiency_min <= 1:
            raise ValueError("efficiency_min must be in (0, 1].")

        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_high"] = data["high"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["prior_close"]).abs(),
            (data["low"] - out["prior_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["prior_atr"] = out["true_range"].rolling(
            params.atr_window, min_periods=params.atr_window
        ).mean().shift(1)
        out["opening_gap"] = data["open"] - out["prior_close"]
        out["gap_atr"] = out["opening_gap"].div(
            out["prior_atr"].where(out["prior_atr"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["close_change"] = data["close"].diff()
        out["path_length"] = out["close_change"].abs().rolling(
            params.trend_window, min_periods=params.trend_window
        ).sum()
        out["net_change"] = out["close_change"].rolling(
            params.trend_window, min_periods=params.trend_window
        ).sum()
        out["efficiency"] = out["net_change"].div(
            out["path_length"].where(out["path_length"] > 0)
        ).replace([np.inf, -np.inf], np.nan)
        out["gap_agreement"] = (
            (out["gap_atr"] >= params.gap_atr_min)
            & (data["open"] > out["prior_high"])
            & (data["low"] > out["prior_high"])
        ).fillna(False)
        out["trend_agreement"] = (
            out["efficiency"] >= params.efficiency_min
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: DisplacementParams,
    ) -> SignalFrame:
        if params.holding_bars < 1:
            raise ValueError("holding_bars must be positive.")
        entries = (
            indicators["gap_agreement"].fillna(False).astype(bool)
            & indicators["trend_agreement"].fillna(False).astype(bool)
        ).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, candidate in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Force a flat bar even if a fresh candidate appears.
                    exit_at = -1
            elif candidate:
                signal[i] = 1
                exit_at = i + params.holding_bars

        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = DisplacementParams()
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({
        "gap_agreement": True,
        "trend_agreement": True,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
    indicators.loc[index[1:5], "trend_agreement"] = False
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:6].tolist() == [0, 1, 1, 1, 1, 0]
    assert (result["size"] > 0).all()

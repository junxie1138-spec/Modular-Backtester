from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    range_lookback: int = 20
    atr_lookback: int = 14
    percentile_lookback: int = 126
    shock_quantile: float = 0.85
    holding_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791195856"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.range_lookback, params.atr_lookback) + params.percentile_lookback + 1)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.range_lookback, params.atr_lookback, params.percentile_lookback) < 2:
            raise ValueError("Lookbacks must be at least two bars")
        if not 0.0 < params.shock_quantile < 1.0:
            raise ValueError("shock_quantile must be between zero and one")
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two")

        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["prior_atr"] = out["true_range"].rolling(
            params.atr_lookback, min_periods=params.atr_lookback
        ).mean().shift(1)
        out["upper"] = data["high"].rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).max().shift(1)
        out["lower"] = data["low"].rolling(
            params.range_lookback, min_periods=params.range_lookback
        ).min().shift(1)
        out["envelope_width"] = (out["upper"] - out["lower"]).where(
            out["upper"] > out["lower"]
        )
        out["relative_position"] = (data["close"] - out["lower"]) / out["envelope_width"]
        out["centered_position"] = 2.0 * out["relative_position"] - 1.0
        out["volatility_shock"] = out["true_range"] / out["prior_atr"].where(out["prior_atr"] > 0.0)
        out["edge_shock"] = (
            out["volatility_shock"] * out["centered_position"].abs()
        ).replace([np.inf, -np.inf], np.nan)
        out["shock_threshold"] = out["edge_shock"].rolling(
            params.percentile_lookback, min_periods=params.percentile_lookback
        ).quantile(params.shock_quantile).shift(1)
        out["contained"] = out["relative_position"].between(0.0, 1.0)
        out["eligible"] = (
            out["contained"]
            & out["edge_shock"].notna()
            & out["shock_threshold"].notna()
            & (out["edge_shock"] > out["shock_threshold"])
        )
        out["entry"] = np.where(
            out["eligible"],
            -np.sign(out["centered_position"].fillna(0.0)),
            0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if params.holding_bars not in (1, 2):
            raise ValueError("holding_bars must be one or two")
        entries = indicators["entry"].fillna(0).astype(int).to_numpy()
        signals = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for bar, entry in enumerate(entries):
            if position != 0:
                if bar >= exit_bar:
                    position = 0
            elif entry != 0:
                position = int(entry)
                exit_bar = bar + params.holding_bars
            signals[bar] = position

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": [1, -1, -1, -1, 1, 1, 0, 0]}, index=index)
    result = strategy.generate_signals(data, indicators, None, StrategyParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0, 0]
    assert (result.data["size"] > 0.0).all()

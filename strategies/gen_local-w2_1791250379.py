from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 5
    area_threshold: float = 0.15
    efficiency_threshold: float = 0.35
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791250379"

    @classmethod
    def params_type(cls) -> type[SignatureParams]:
        return SignatureParams

    @staticmethod
    def warmup_bars(params: SignatureParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        return int(params.lookback + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        length = params.lookback
        returns = np.log(data["close"].where(data["close"] > 0)).diff()
        returns = returns.replace([np.inf, -np.inf], np.nan)
        net_return = returns.rolling(length, min_periods=length).sum()
        variation = returns.abs().rolling(length, min_periods=length).sum()
        denominator = variation.where(variation > 0)
        area = pd.Series(0.0, index=data.index)
        # Antisymmetric time-price signature: later increments receive positive weights.
        for lag in range(length):
            area = area + (length - 1 - 2 * lag) * returns.shift(lag)
        result = pd.DataFrame(index=data.index)
        result["net_return"] = net_return
        result["variation"] = variation
        result["signature_area"] = area / ((length - 1) * denominator)
        result["efficiency"] = net_return / denominator
        return result.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: SignatureParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not 0 <= params.area_threshold <= 1:
            raise ValueError("area_threshold must be between 0 and 1")
        if not 0 <= params.efficiency_threshold <= 1:
            raise ValueError("efficiency_threshold must be between 0 and 1")
        eligible = (
            indicators["signature_area"].gt(params.area_threshold)
            & indicators["efficiency"].gt(params.efficiency_threshold)
            & indicators["net_return"].gt(0)
        ).fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        entry_bar = -1
        for bar in range(len(data)):
            if entry_bar >= 0:
                if bar - entry_bar >= params.hold_bars:
                    entry_bar = -1
                else:
                    signals[bar] = 1
            elif eligible[bar]:
                entry_bar = bar
                signals[bar] = 1
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    sample = pd.DataFrame({name: np.ones(7) for name in ("open", "high", "low", "close", "volume")}, index=index)
    evidence = pd.DataFrame({"signature_area": 1.0, "efficiency": 1.0, "net_return": 0.01}, index=index)
    for holding, expected in ((1, [0, 1, 0, 1, 0, 1, 0]), (2, [0, 1, 1, 0, 1, 1, 0])):
        result = GeneratedStrategy.generate_signals(sample, evidence, None, SignatureParams(hold_bars=holding))
        assert result.data["signal"].tolist() == expected
    flat = GeneratedStrategy.indicators(sample, SignatureParams())
    result = GeneratedStrategy.generate_signals(sample, flat, None, SignatureParams())
    assert result.data["signal"].eq(0).all()

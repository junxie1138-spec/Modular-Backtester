from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class CounterflowParams:
    lookback: int = 6
    counterflow_threshold: float = 0.35
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[CounterflowParams]):
    strategy_id = "gen_local_1790964948"

    @classmethod
    def params_type(cls) -> type[CounterflowParams]:
        return CounterflowParams

    def warmup_bars(self, params: CounterflowParams) -> int:
        if not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be between 2 and 9")
        if not 0.0 < params.counterflow_threshold < 1.0:
            raise ValueError("counterflow_threshold must be between 0 and 1")
        if not 5 <= params.hold_bars <= 10:
            raise ValueError("hold_bars must be between 5 and 10")
        return int(params.lookback + 1)

    def indicators(self, data: pd.DataFrame, params: CounterflowParams) -> pd.DataFrame:
        self.warmup_bars(params)
        previous_close = data["close"].shift(1).where(lambda s: s > 0)
        returns = (data["close"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        gaps = (data["open"] / previous_close - 1.0).replace([np.inf, -np.inf], np.nan)
        length = params.lookback
        return_energy = returns.pow(2).rolling(length, min_periods=length).mean()
        gap_energy = gaps.pow(2).rolling(length, min_periods=length).mean()
        denominator = np.sqrt(return_energy * gap_energy).replace(0.0, np.nan)
        counterflow = -(returns * gaps).rolling(length, min_periods=length).mean() / denominator
        return pd.DataFrame({
            "close_return": returns,
            "gap": gaps,
            "return_energy": return_energy,
            "gap_energy": gap_energy,
            "counterflow": counterflow.replace([np.inf, -np.inf], np.nan),
            "net_return": returns.rolling(length, min_periods=length).sum(),
            "net_gap": gaps.rolling(length, min_periods=length).sum(),
        }, index=data.index)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: CounterflowParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = (
            indicators.notna().all(axis=1)
            & (indicators["counterflow"] >= params.counterflow_threshold)
            & (indicators["net_return"] > 0.0)
            & (indicators["net_gap"] < 0.0)
            & (indicators["close_return"] > 0.0)
            & (indicators["gap"] < 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for i, enter in enumerate(entries):
            if exit_bar >= 0:
                if i == exit_bar:
                    exit_bar = -1
                else:
                    signal[i] = 1
            elif enter:
                signal[i] = 1
                exit_bar = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = CounterflowParams(lookback=2, hold_bars=5)
    data = pd.DataFrame(index=pd.date_range("2020-01-01", periods=14))
    indicators = pd.DataFrame({
        "close_return": 0.01, "gap": -0.005,
        "return_energy": 0.0001, "gap_energy": 0.000025,
        "counterflow": 1.0, "net_return": 0.02, "net_gap": -0.01,
    }, index=data.index)
    indicators.iloc[:2] = np.nan
    strategy = GeneratedStrategy()
    result = strategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(CounterflowParams(lookback=9)) == 10

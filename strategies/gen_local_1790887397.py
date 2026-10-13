from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    capture_window: int = 5
    atr_window: int = 20
    expansion_threshold: float = 1.05
    capture_threshold: float = 0.15
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790887397"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return max(200, params.capture_window + 1, params.atr_window + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if params.capture_window < 1 or params.atr_window < 1:
            raise ValueError("Lookback windows must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr_fast"] = out["true_range"].rolling(params.capture_window).mean()
        out["atr_slow"] = out["true_range"].rolling(params.atr_window).mean()
        out["atr_expansion"] = out["atr_fast"] / out["atr_slow"].replace(0.0, np.nan)
        out["low_change"] = data["low"].diff()
        out["floor_advance"] = out["low_change"].rolling(params.capture_window).sum()
        out["range_budget"] = out["true_range"].rolling(params.capture_window).sum()
        out["floor_capture"] = out["floor_advance"] / out["range_budget"].replace(0.0, np.nan)
        out["higher_low"] = out["low_change"].gt(0).astype(float).where(out["low_change"].notna())
        out["higher_low_fraction"] = out["higher_low"].rolling(params.capture_window).mean()
        out["ma200"] = data["close"].rolling(200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StrategyParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        required = indicators[["ma200", "atr_expansion", "floor_capture", "higher_low_fraction"]]
        ready = required.notna().all(axis=1)
        eligible = (
            ready
            & data["close"].gt(indicators["ma200"])
            & indicators["atr_expansion"].ge(params.expansion_threshold)
            & indicators["floor_capture"].ge(params.capture_threshold)
            & indicators["higher_low_fraction"].ge(0.8)
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_bar = -1
        for bar in range(len(data)):
            if exit_bar >= 0:
                if bar < exit_bar:
                    signal[bar] = 1
                else:
                    exit_bar = -1
            elif eligible[bar]:
                signal[bar] = 1
                exit_bar = bar + params.hold_bars
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=11, freq="B")
    sample = pd.DataFrame({"close": 101.0}, index=index)
    features = pd.DataFrame({
        "ma200": 100.0,
        "atr_expansion": 1.2,
        "floor_capture": 0.3,
        "higher_low_fraction": 1.0,
    }, index=index)
    result = GeneratedStrategy.generate_signals(sample, features, None, StrategyParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0]
    features["ma200"] = np.nan
    result = GeneratedStrategy.generate_signals(sample, features, None, StrategyParams())
    assert result.data["signal"].eq(0).all()

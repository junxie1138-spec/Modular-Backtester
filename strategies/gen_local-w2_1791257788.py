from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class SignatureParams:
    lookback: int = 6
    area_threshold: float = 0.15
    efficiency_threshold: float = 0.45
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[SignatureParams]):
    strategy_id = "gen_local-w2_1791257788"

    @classmethod
    def params_type(cls):
        return SignatureParams

    def warmup_bars(self, params: SignatureParams) -> int:
        if not isinstance(params.lookback, int) or not 2 <= params.lookback <= 9:
            raise ValueError("lookback must be an integer from 2 through 9")
        if not isinstance(params.hold_bars, int) or params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        for value in (params.area_threshold, params.efficiency_threshold):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("thresholds must be finite and between 0 and 1")
        return params.lookback + 1

    def indicators(self, data: pd.DataFrame, params: SignatureParams) -> pd.DataFrame:
        self.warmup_bars(params)
        length = params.lookback
        close = data["close"].astype(float)
        returns = np.log(close.where(close > 0)).diff().replace([np.inf, -np.inf], np.nan)
        net = returns.rolling(length, min_periods=length).sum()
        path = returns.abs().rolling(length, min_periods=length).sum()
        # Twice the antisymmetric level-two time-price signature area.
        area = pd.Series(0.0, index=data.index)
        for lag in range(length):
            area = area + (length - 1 - 2 * lag) * returns.shift(lag)
        denominator = path.where(path > 0)
        result = pd.DataFrame(index=data.index)
        result["log_return"] = returns.fillna(0.0)
        result["net_return"] = net.fillna(0.0)
        result["path_length"] = path.fillna(0.0)
        result["signature_area"] = (area / ((length - 1) * denominator)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["efficiency"] = (net / denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        result["ready"] = net.notna() & path.gt(0)
        return result

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: SignatureParams) -> SignalFrame:
        self.warmup_bars(params)
        area = indicators["signature_area"]
        efficiency = indicators["efficiency"]
        ready = indicators["ready"]
        long_entry = ready & area.gt(params.area_threshold) & efficiency.gt(params.efficiency_threshold)
        short_entry = ready & area.lt(-params.area_threshold) & efficiency.lt(-params.efficiency_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for bar in range(len(data)):
            if position != 0:
                if bar - entry_bar >= params.hold_bars:
                    position = 0
            elif entries[bar] != 0:
                position = int(entries[bar])
                entry_bar = bar
            signals[bar] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy.__new__(GeneratedStrategy)
    index = pd.date_range("2020-01-01", periods=20)
    params = SignatureParams(area_threshold=0.0, efficiency_threshold=0.0)
    for direction in (1, -1):
        close = 100.0 * np.exp(direction * np.cumsum(np.arange(20) * 0.001))
        sample = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=index)
        derived = strategy.indicators(sample, params)
        output = strategy.generate_signals(sample, derived, None, params).data
        assert output["signal"].iloc[:7].eq(0).all()
        assert output["signal"].iloc[7:10].tolist() == [direction, direction, 0]
        assert strategy.warmup_bars(params) <= 10

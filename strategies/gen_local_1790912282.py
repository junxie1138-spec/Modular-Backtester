from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class DistributedMomentumParams:
    momentum_window: int = 15
    percentile_window: int = 126
    entry_percentile: float = 0.80
    atr_window: int = 14
    trailing_atr: float = 3.0
    holding_bars: int = 18


class GeneratedStrategy(BaseStrategy[DistributedMomentumParams]):
    strategy_id = "gen_local_1790912282"

    @classmethod
    def params_type(cls) -> type[DistributedMomentumParams]:
        return DistributedMomentumParams

    def warmup_bars(self, params: DistributedMomentumParams) -> int:
        return int(max(params.momentum_window + params.percentile_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: DistributedMomentumParams) -> pd.DataFrame:
        if min(params.momentum_window, params.percentile_window, params.atr_window, params.holding_bars) < 1:
            raise ValueError("Window lengths and holding_bars must be positive")
        if not 0.0 < params.entry_percentile < 1.0:
            raise ValueError("entry_percentile must be between zero and one")
        if not np.isfinite(params.trailing_atr) or params.trailing_atr <= 0.0:
            raise ValueError("trailing_atr must be finite and positive")

        out = pd.DataFrame(index=data.index)
        returns = data["close"].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        positive = returns.clip(lower=0.0)
        window = params.momentum_window
        positive_sum = positive.rolling(window, min_periods=window).sum()
        positive_energy = positive.pow(2).rolling(window, min_periods=window).sum()
        absolute_sum = returns.abs().rolling(window, min_periods=window).sum()
        net_return = returns.rolling(window, min_periods=window).sum()
        effective_days = positive_sum.pow(2).div(positive_energy.where(positive_energy > 0.0))
        efficiency = net_return.div(absolute_sum.where(absolute_sum > 0.0))
        score = (effective_days / window * efficiency).replace([np.inf, -np.inf], np.nan)
        threshold = score.shift(1).rolling(params.percentile_window, min_periods=params.percentile_window).quantile(params.entry_percentile)

        previous_close = data["close"].shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        out["returns"] = returns
        out["positive_sum"] = positive_sum
        out["positive_energy"] = positive_energy
        out["absolute_sum"] = absolute_sum
        out["net_return"] = net_return
        out["effective_days"] = effective_days
        out["efficiency"] = efficiency
        out["score"] = score
        out["threshold"] = threshold
        out["true_range"] = true_range
        out["atr"] = atr
        out["entry"] = (score > threshold) & (net_return > 0.0) & (atr > 0.0) & data["close"].gt(0.0)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: DistributedMomentumParams) -> SignalFrame:
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        active = False
        age = 0
        high_water = -np.inf
        stop = -np.inf

        for i in range(len(data)):
            close = closes[i]
            atr = atrs[i]
            valid = np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0
            if active:
                age += 1
                # The first in-trade close is the bar after the entry decision.
                if valid:
                    high_water = max(high_water, close)
                    stop = max(stop, high_water - params.trailing_atr * atr)
                if age >= params.holding_bars or (valid and close <= stop):
                    active = False
                    continue
                signals[i] = 1
            elif valid and entries[i]:
                active = True
                age = 0
                high_water = -np.inf
                stop = -np.inf
                signals[i] = 1

        df = pd.DataFrame(index=data.index)
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": [100, 100, 105, 110, 108, 106, 105, 105], "high": [101, 101, 106, 111, 109, 107, 106, 106], "low": [99, 99, 104, 109, 107, 105, 104, 104], "close": [100, 100, 105, 110, 108, 106, 105, 105], "volume": 1000}, index=index)
    derived = pd.DataFrame({"atr": [2, 2, 2, 2, 4, 4, 4, 4], "entry": [True, False, False, False, False, False, False, False]}, index=index)
    strategy = GeneratedStrategy()
    params = DistributedMomentumParams(trailing_atr=1.0)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    params.holding_bars = 2
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    assert result["size"].gt(0.0).all()
    assert strategy.warmup_bars(params) >= params.momentum_window + params.percentile_window + 1


if __name__ == "__main__":
    _self_check()

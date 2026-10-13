from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    fractional_order: float = 0.5
    compression_ratio: float = 0.7
    thrust_threshold: float = 0.25
    profit_target: float = 0.025
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791256770"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if not 0.0 < params.fractional_order < 1.0:
            raise ValueError("fractional_order must be between zero and one")
        if not 0.0 < params.compression_ratio < 1.0:
            raise ValueError("compression_ratio must be between zero and one")
        if not np.isfinite(params.thrust_threshold) or params.thrust_threshold <= 0:
            raise ValueError("thrust_threshold must be finite and positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if not isinstance(params.max_hold_bars, int) or not 5 <= params.max_hold_bars <= 10:
            raise ValueError("max_hold_bars must be an integer from five to ten")
        return 7

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float)
        weights = [1.0]
        for k in range(1, 4):
            weights.append(-weights[-1] * (params.fractional_order - k + 1) / k)
        fractional_close = sum(weights[k] * close.shift(k) for k in range(4))
        # ponytail: four-tap fractional memory; extend only if a longer warmup is allowed.
        thrust = fractional_close - sum(weights) * close.shift(3)
        ceiling = data["high"].rolling(3, min_periods=3).max()
        floor = data["low"].rolling(3, min_periods=3).min()
        width = (ceiling - floor).where(ceiling > floor)
        prior_width = width.shift(3)
        pressure = (thrust / width).replace([np.inf, -np.inf], np.nan)
        compressed = width.le(params.compression_ratio * prior_width)
        entry = (
            compressed
            & pressure.gt(params.thrust_threshold)
            & pressure.shift(1).le(params.thrust_threshold)
            & close.lt(ceiling)
            & close.gt(0)
        ).fillna(False)
        return pd.DataFrame({
            "fractional_close": fractional_close,
            "fractional_thrust": thrust,
            "ceiling": ceiling,
            "floor": floor,
            "width": width,
            "prior_width": prior_width,
            "pressure": pressure,
            "compressed": compressed.fillna(False),
            "entry": entry,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: Params,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        entry_bar = -1
        entry_price = np.nan
        pending = False
        for i in range(len(data)):
            # An entry ordered at the previous close fills at this bar's open.
            if pending:
                entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    raise ValueError("entry open must be finite and positive")
                entry_bar = i
                pending = False
            if entry_bar >= 0:
                profit_exit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1 + params.profit_target)
                time_exit = i - entry_bar + 1 >= params.max_hold_bars
                if profit_exit or time_exit:
                    entry_bar = -1
                    entry_price = np.nan
                else:
                    signal[i] = 1
            elif i >= warmup - 1 and entries[i]:
                signal[i] = 1
                pending = True
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    index = pd.date_range("2020-01-01", periods=20)
    data = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    derived = strategy.indicators(data, params)
    assert not derived["entry"].any()
    derived["entry"] = False
    derived.iloc[6, derived.columns.get_loc("entry")] = True
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].tolist() == [0] * 7 + [1] * 8 + [0] * 5
    data.iloc[7, data.columns.get_loc("open")] = 102.0
    data.iloc[7, data.columns.get_loc("close")] = 103.0
    data.iloc[8, data.columns.get_loc("close")] = 105.0
    signals = strategy.generate_signals(data, derived, None, params).data
    assert signals["signal"].tolist() == [0] * 7 + [1] * 2 + [0] * 11
    assert signals["size"].gt(0).all()
    assert strategy.warmup_bars(params) <= 10


if __name__ == "__main__":
    _self_check()

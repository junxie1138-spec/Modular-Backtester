from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RecoilParams:
    fractional_order: float = 0.45
    fractional_terms: int = 16
    position_window: int = 20
    recovery_ceiling: float = 0.45
    profit_target: float = 0.008
    max_hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RecoilParams]):
    strategy_id = "gen_local_1791344873"

    @classmethod
    def params_type(cls):
        return RecoilParams

    def warmup_bars(self, params: RecoilParams) -> int:
        return int(params.fractional_terms + params.position_window)

    def indicators(self, data: pd.DataFrame, params: RecoilParams) -> pd.DataFrame:
        if not (0.0 < params.fractional_order < 1.0):
            raise ValueError("fractional_order must be between zero and one")
        if params.fractional_terms < 2 or params.position_window < 2:
            raise ValueError("lookbacks must be at least two")
        if not (0.0 < params.recovery_ceiling < 1.0):
            raise ValueError("recovery_ceiling must be between zero and one")
        if params.profit_target <= 0.0 or params.max_hold_bars not in (1, 2):
            raise ValueError("positive target and one or two holding bars required")
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        # ponytail: finite fractional memory; increase fractional_terms for a longer tail.
        fractional = close.copy()
        weight = 1.0
        for lag in range(1, params.fractional_terms):
            weight *= (lag - 1.0 - params.fractional_order) / lag
            fractional = fractional + weight * close.shift(lag)
        floor = fractional.rolling(params.position_window).min()
        ceiling = fractional.rolling(params.position_window).max()
        span = (ceiling - floor).replace(0.0, np.nan)
        position = (fractional - floor) / span
        baseline = close.rolling(params.position_window).mean()
        rising = fractional.diff().gt(0.0) & position.diff().gt(0.0)
        eligible = (
            rising
            & position.gt(0.0)
            & position.lt(params.recovery_ceiling)
            & close.lt(baseline)
            & close.gt(0.0)
        ).fillna(False)
        return pd.DataFrame({
            "fractional_close": fractional,
            "fractional_floor": floor,
            "fractional_ceiling": ceiling,
            "fractional_span": span,
            "relative_position": position,
            "close_baseline": baseline,
            "rising": rising.fillna(False),
            "eligible": eligible,
        }, index=data.index)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RecoilParams,
    ) -> SignalFrame:
        eligible = indicators["eligible"].fillna(False).astype(bool)
        confirmed = (eligible & eligible.shift(1, fill_value=False)).to_numpy()
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        age = 0
        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this bar's open.
                if age == 0:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0.0:
                        active = False
                        continue
                age += 1
                reached_target = (
                    np.isfinite(closes[i])
                    and closes[i] >= entry_price * (1.0 + params.profit_target)
                )
                if reached_target or age >= params.max_hold_bars:
                    active = False
                    age = 0
                    continue
            elif confirmed[i]:
                active = True
                entry_price = np.nan
                age = 0
            signals[i] = int(active)
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RecoilParams()
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({
        "open": 100.0, "high": 103.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    indicators = pd.DataFrame({
        "eligible": [True, True, False, False, False, False, False, False]
    }, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 1, 0, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]
    flat = GeneratedStrategy.indicators(None, data.assign(close=100.0), params)
    assert not flat["eligible"].any()
    assert result["size"].gt(0.0).all()


if __name__ == "__main__":
    _self_check()

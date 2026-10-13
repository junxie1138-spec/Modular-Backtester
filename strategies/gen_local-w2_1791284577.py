from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    complexity_ceiling: float = 0.25
    trend_floor: float = 0.5
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local-w2_1791284577"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return 8

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        a, b, c = data.close.shift(2), data.close.shift(1), data.close
        valid = (
            np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            & (a != b) & (b != c) & (a != c)
        )
        pattern = (
            (b > a).astype(int)
            + 2 * (c > b).astype(int)
            + 4 * (c > a).astype(int)
        ).astype(float).where(valid)
        recurrence = pattern.eq(pattern.shift(2)).astype(float).where(
            pattern.notna() & pattern.shift(2).notna()
        )
        vote = (
            pattern.eq(7).astype(float) - pattern.eq(0).astype(float)
        ).where(pattern.notna())
        return pd.DataFrame({
            "ordinal_pattern": pattern,
            "recurrence": recurrence,
            "complexity": 1.0 - recurrence.rolling(4, min_periods=4).mean(),
            "monotone_vote": vote,
            "trend_strength": vote.rolling(6, min_periods=6).mean(),
        }, index=data.index)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: OrdinalParams,
    ) -> SignalFrame:
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not (
            0 <= params.complexity_ceiling <= 1
            and 0 < params.trend_floor <= 1
        ):
            raise ValueError("Invalid complexity_ceiling or trend_floor")
        strength = indicators["trend_strength"]
        ready = (
            indicators["complexity"].le(params.complexity_ceiling)
            & strength.abs().ge(params.trend_floor)
        )
        direction = np.sign(strength.fillna(0)).astype(int)
        candidate = direction.where(
            ready & indicators["monotone_vote"].eq(direction), 0
        ).to_numpy()
        signal = np.zeros(len(data), dtype=int)
        position, exit_bar = 0, -1
        for i, entry in enumerate(candidate):
            if position:
                if i == exit_bar:
                    position = 0
                else:
                    signal[i] = position
            elif entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
                signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = OrdinalParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=8)
    for side in (1, -1):
        close = 100.0 + side * np.arange(20)
        data = pd.DataFrame(
            {name: close for name in ("open", "high", "low", "close")},
            index=pd.date_range("2020-01-01", periods=20),
        )
        data["volume"] = 1000.0
        result = strategy.generate_signals(
            data, strategy.indicators(data, params), ctx, params
        ).data
        assert result.signal.tolist() == [0] * 8 + [side, side, 0] * 4
        one = OrdinalParams(hold_bars=1)
        result_one = strategy.generate_signals(
            data, strategy.indicators(data, one), ctx, one
        ).data
        assert result_one.signal.tolist() == [0] * 8 + [side, 0] * 6
        altered = data.copy()
        altered.loc[altered.index[15]:, "close"] = 1.0
        changed = strategy.generate_signals(
            altered, strategy.indicators(altered, params), ctx, params
        ).data
        assert changed.iloc[:16].equals(result.iloc[:16])
    data["close"] = 100.0
    assert not strategy.generate_signals(
        data, strategy.indicators(data, params), ctx, params
    ).data.signal.any()

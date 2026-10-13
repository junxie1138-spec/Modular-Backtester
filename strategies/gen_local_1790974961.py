from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalPositionParams:
    complexity_min: float = 0.72
    position_max: float = 0.40
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalPositionParams]):
    strategy_id = "gen_local_1790974961"

    @classmethod
    def params_type(cls) -> type[OrdinalPositionParams]:
        return OrdinalPositionParams

    @staticmethod
    def warmup_bars(params: OrdinalPositionParams) -> int:
        return 10

    @staticmethod
    def indicators(data: pd.DataFrame, params: OrdinalPositionParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["range_high"] = data["high"].rolling(4, min_periods=4).max()
        out["range_low"] = data["low"].rolling(4, min_periods=4).min()
        out["range_span"] = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["position"] = ((data["close"] - out["range_low"]) / out["range_span"]).replace(
            [np.inf, -np.inf], np.nan
        )
        out["position_lag1"] = out["position"].shift(1)
        out["position_lag2"] = out["position"].shift(2)
        a, b, c = out["position_lag2"], out["position_lag1"], out["position"]
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        out["pattern"] = (
            a.lt(b).astype(int) + 2 * a.lt(c).astype(int) + 4 * b.lt(c).astype(int)
        ).where(valid)
        out["valid_patterns"] = valid.astype(float).rolling(5, min_periods=5).sum()
        collision = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            column = "frequency_" + str(code)
            out[column] = out["pattern"].eq(code).astype(float).rolling(5, min_periods=5).mean()
            collision = collision + out[column].pow(2)
        out["complexity"] = ((1.0 - collision) * 6.0 / 5.0).where(
            out["valid_patterns"].eq(5.0)
        )
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: OrdinalPositionParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not (0.0 <= params.complexity_min <= 1.0 and 0.0 <= params.position_max <= 1.0):
            raise ValueError("complexity_min and position_max must be between 0 and 1")
        ready = (
            indicators["complexity"].ge(params.complexity_min)
            & indicators["position"].le(params.position_max)
            & indicators["position"].gt(indicators["position_lag1"])
        ).fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i in range(len(data)):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    exit_at = -1
            elif ready[i]:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=16)
    data = pd.DataFrame(index=index)
    forced = pd.DataFrame(
        {"complexity": 0.96, "position": 0.25, "position_lag1": 0.20}, index=index
    )
    for hold in (3, 4, 5):
        params = OrdinalPositionParams(hold_bars=hold)
        result = GeneratedStrategy.generate_signals(data, forced, None, params).data
        expected = [0] + [int(i % (hold + 1) < hold) for i in range(len(data) - 1)]
        assert result["signal"].tolist() == expected
        forced_once = forced.copy()
        forced_once.loc[index[1]:, "complexity"] = 0.0
        result = GeneratedStrategy.generate_signals(data, forced_once, None, params).data
        assert result["signal"].tolist() == [0] + [1] * hold + [0] * (len(data) - hold - 1)


if __name__ == "__main__":
    _self_check()

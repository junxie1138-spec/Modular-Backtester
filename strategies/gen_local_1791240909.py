from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeComplexityParams:
    complexity_window: int = 18
    reference_window: int = 126
    complexity_quantile: float = 0.25
    shock_quantile: float = 0.10
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[RangeComplexityParams]):
    strategy_id = "gen_local_1791240909"

    @classmethod
    def params_type(cls) -> type[RangeComplexityParams]:
        return RangeComplexityParams

    def warmup_bars(self, params: RangeComplexityParams) -> int:
        if params.complexity_window < 3 or params.reference_window < 20:
            raise ValueError("Lookback windows are too short")
        if not 0.0 < params.complexity_quantile < 1.0:
            raise ValueError("complexity_quantile must be between zero and one")
        if not 0.0 < params.shock_quantile < 1.0:
            raise ValueError("shock_quantile must be between zero and one")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two")
        return int(params.complexity_window + params.reference_window + 3)

    def indicators(self, data: pd.DataFrame, params: RangeComplexityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        out["relative_range"] = ((data["high"] - data["low"]) / close).replace([np.inf, -np.inf], np.nan)
        out["relative_range"] = out["relative_range"].where(out["relative_range"] >= 0)
        out["range_lag2"] = out["relative_range"].shift(2)
        out["range_lag1"] = out["relative_range"].shift(1)
        a = out["range_lag2"]
        b = out["range_lag1"]
        c = out["relative_range"]
        valid = a.notna() & b.notna() & c.notna()
        # Strict comparisons break ties by chronological order.
        out["ordinal_code"] = ((a > b).astype(int) * 4 + (a > c).astype(int) * 2 + (b > c).astype(int)).where(valid)
        probability_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            column = f"pattern_probability_{code}"
            out[column] = out["ordinal_code"].eq(code).astype(float).where(valid).rolling(params.complexity_window, min_periods=params.complexity_window).mean()
            probability_columns.append(column)
        probabilities = out[probability_columns]
        out["complexity"] = -(probabilities * np.log(probabilities.where(probabilities > 0, 1.0))).sum(axis=1, min_count=6) / np.log(6.0)
        out["complexity_threshold"] = out["complexity"].shift(1).rolling(params.reference_window, min_periods=params.reference_window).quantile(params.complexity_quantile)
        out["two_day_return"] = (close / close.shift(2) - 1.0).replace([np.inf, -np.inf], np.nan)
        out["shock_threshold"] = out["two_day_return"].shift(1).rolling(params.reference_window, min_periods=params.reference_window).quantile(params.shock_quantile)
        out["contracting_ranges"] = valid & (a > b) & (b > c)
        out["entry"] = (
            out["contracting_ranges"]
            & (out["complexity"] <= out["complexity_threshold"])
            & (out["two_day_return"] <= out["shock_threshold"])
            & (out["two_day_return"] < 0)
            & (close < close.shift(1))
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: RangeComplexityParams) -> SignalFrame:
        self.warmup_bars(params)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=np.int64)
        exit_at = -1
        for i, entry in enumerate(entries):
            if exit_at >= 0:
                if i < exit_at:
                    signal[i] = 1
                else:
                    # Emit a flat bar even if a new entry qualifies here.
                    exit_at = -1
            elif entry:
                signal[i] = 1
                exit_at = i + params.hold_bars
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    for hold in (1, 2):
        params = RangeComplexityParams(hold_bars=hold)
        for entries in ([True] + [False] * 9, [True] * 10):
            indicators = pd.DataFrame({"entry": entries}, index=index)
            result = strategy.generate_signals(data, indicators, None, params).data
            assert result["signal"].iloc[0] == 0
            assert result["signal"].iloc[1:hold + 1].eq(1).all()
            assert result["signal"].iloc[hold + 1] == 0
            assert result["signal"].isin([0, 1]).all()
            assert result["size"].gt(0).all()
    indicators = strategy.indicators(data, RangeComplexityParams())
    assert not indicators["entry"].any()


if __name__ == "__main__":
    _self_check()

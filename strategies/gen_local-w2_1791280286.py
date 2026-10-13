from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalRecruitmentParams:
    pattern_window: int = 5
    min_complexity: float = 0.5
    entry_ceiling: float = 0.45
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalRecruitmentParams]):
    strategy_id = "gen_local-w2_1791280286"

    @classmethod
    def params_type(cls):
        return OrdinalRecruitmentParams

    def warmup_bars(self, params: OrdinalRecruitmentParams) -> int:
        if not 3 <= params.pattern_window <= 7:
            raise ValueError("pattern_window must be between 3 and 7")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be 1 or 2")
        if not 0.0 <= params.min_complexity <= 1.0:
            raise ValueError("min_complexity must be between 0 and 1")
        if not 0.0 <= params.entry_ceiling <= 1.0:
            raise ValueError("entry_ceiling must be between 0 and 1")
        return int(max(6, params.pattern_window + 3))

    def indicators(self, data: pd.DataFrame, params: OrdinalRecruitmentParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(self, params)
        out = pd.DataFrame(index=data.index)
        span = (data["high"] - data["low"]).where(lambda x: x > 0)
        location = ((data["close"] - data["low"]) / span).replace([np.inf, -np.inf], np.nan)
        location = location.where(location.between(0.0, 1.0))
        out["close_location"] = location
        a, b, c = location.shift(2), location.shift(1), location
        valid = a.notna() & b.notna() & c.notna() & a.ne(b) & a.ne(c) & b.ne(c)
        pattern = (4 * a.lt(b).astype(int) + 2 * a.lt(c).astype(int) + b.lt(c).astype(int)).astype(float)
        out["ordinal_pattern"] = pattern.where(valid)
        previous = out["ordinal_pattern"].shift(1)
        out["valid_history"] = previous.notna().astype(float).rolling(params.pattern_window, min_periods=params.pattern_window).sum()
        count_columns = []
        for code in (0, 1, 3, 4, 6, 7):
            name = f"pattern_count_{code}"
            out[name] = previous.eq(code).astype(float).rolling(params.pattern_window, min_periods=params.pattern_window).sum()
            count_columns.append(name)
        ready = out["valid_history"].eq(params.pattern_window)
        out["complexity"] = (out[count_columns].gt(0).sum(axis=1) / 6.0).where(ready)
        out["range_low"] = data["low"].rolling(6, min_periods=6).min()
        out["range_high"] = data["high"].rolling(6, min_periods=6).max()
        envelope = (out["range_high"] - out["range_low"]).where(lambda x: x > 0)
        out["range_position"] = ((data["close"] - out["range_low"]) / envelope).replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            ready
            & out["ordinal_pattern"].eq(7)
            & out["pattern_count_7"].eq(0)
            & out["complexity"].ge(params.min_complexity)
            & out["range_position"].between(0.0, params.entry_ceiling)
        ).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: OrdinalRecruitmentParams) -> SignalFrame:
        GeneratedStrategy.warmup_bars(self, params)
        signal = np.zeros(len(data), dtype=int)
        exit_at = -1
        for i, candidate in enumerate(indicators["entry"].fillna(False).to_numpy(dtype=bool)):
            if i < exit_at:
                signal[i] = 1
            elif i > exit_at and candidate:
                signal[i] = 1
                exit_at = i + params.hold_bars
            # At exit_at, remain flat even if another entry qualifies.
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(index=index)
    indicators = pd.DataFrame({"entry": True}, index=index)
    params = OrdinalRecruitmentParams()
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 0]
    params.hold_bars = 1
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params)
    assert result.data["signal"].tolist() == [0, 1, 0, 1, 0, 1, 0]
    params.pattern_window = 7
    assert GeneratedStrategy.warmup_bars(None, params) == 10

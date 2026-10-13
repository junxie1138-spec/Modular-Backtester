from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MidmonthParams:
    tail_fraction: float = 0.15
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[MidmonthParams]):
    strategy_id = "gen_local_1791286341"

    @classmethod
    def params_type(cls) -> type[MidmonthParams]:
        return MidmonthParams

    def warmup_bars(self, params: MidmonthParams) -> int:
        return 130

    def indicators(self, data: pd.DataFrame, params: MidmonthParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["body"] = (data["close"] - data["open"]).replace([np.inf, -np.inf], np.nan)
        out["range"] = (data["high"] - data["low"]).where(data["high"] > data["low"])
        out["body_sum"] = out["body"].rolling(5, min_periods=5).sum()
        out["range_sum"] = out["range"].rolling(5, min_periods=5).sum()
        out["pressure"] = (out["body_sum"] / out["range_sum"]).replace([np.inf, -np.inf], np.nan)
        out["pressure_rank"] = out["pressure"].rolling(126, min_periods=126).rank(pct=True)
        out["day"] = data.index.day
        out["month_key"] = data.index.year * 12 + data.index.month
        out["previous_day"] = out["day"].shift(1)
        out["previous_month"] = out["month_key"].shift(1)
        out["midmonth_turn"] = (
            out["day"].between(16, 20)
            & out["previous_day"].lt(16)
            & out["month_key"].eq(out["previous_month"])
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: MidmonthParams,
    ) -> SignalFrame:
        if not np.isfinite(params.tail_fraction) or not 0.0 < params.tail_fraction < 0.5:
            raise ValueError("tail_fraction must be between zero and one half")
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        ranks = indicators["pressure_rank"].to_numpy(dtype=float)
        eligible = indicators["midmonth_turn"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i in range(len(data)):
            if position:
                if i >= exit_at:
                    position = 0
                signal[i] = position
                continue
            if not eligible[i] or not np.isfinite(ranks[i]):
                continue
            if ranks[i] <= params.tail_fraction:
                position = 1
            elif ranks[i] >= 1.0 - params.tail_fraction:
                position = -1
            if position:
                exit_at = i + params.hold_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=23)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"pressure_rank": np.where(np.arange(23) < 9, 0.01, 0.99), "midmonth_turn": True}, index=index)
    result = GeneratedStrategy().generate_signals(data, indicators, None, MidmonthParams()).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:9].eq(1).all()
    assert result["signal"].iloc[9] == 0
    assert result["signal"].iloc[10:18].eq(-1).all()
    assert result["signal"].iloc[18] == 0
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapDebtParams:
    repayment_fraction: float = 0.5
    holding_bars: int = 8


class GeneratedStrategy(BaseStrategy[GapDebtParams]):
    strategy_id = "gen_local-w2_1791322064"

    @classmethod
    def params_type(cls) -> type[GapDebtParams]:
        return GapDebtParams

    @staticmethod
    def warmup_bars(params: GapDebtParams) -> int:
        if not 0.0 < params.repayment_fraction <= 1.0:
            raise ValueError("repayment_fraction must be in (0, 1]")
        if not 5 <= params.holding_bars <= 10:
            raise ValueError("holding_bars must be between 5 and 10")
        return 21

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapDebtParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        out["prior_close"] = data["close"].shift(1)
        out["prior_peak"] = data["close"].rolling(20, min_periods=20).max().shift(1)
        out["gap"] = data["open"] - out["prior_close"]
        out["gap_debt"] = (-out["gap"]).clip(lower=0).rolling(5, min_periods=5).sum().shift(1)
        out["drawdown"] = out["prior_close"] / out["prior_peak"].replace(0, np.nan) - 1.0
        out["repayment"] = out["gap"] / out["gap_debt"].replace(0, np.nan)
        out["retention"] = (data["close"] - out["prior_close"]) / out["gap"].replace(0, np.nan)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["eligible"] = (
            (out["prior_close"] > 0)
            & (out["drawdown"] <= -0.03)
            & (out["gap_debt"] >= 0.01 * out["prior_close"])
            & (out["gap"] > 0)
            & (out["repayment"] >= params.repayment_fraction)
        ).fillna(False)
        out["long_entry"] = (out["eligible"] & (out["retention"] >= 0.5)).fillna(False)
        out["short_entry"] = (out["eligible"] & (out["retention"] <= 0.0)).fillna(False)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GapDebtParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        long_entry = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        short_entry = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i in range(len(data)):
            if position != 0:
                if i == exit_bar:
                    position = 0
                signal[i] = position
                continue
            if long_entry[i]:
                position = 1
            elif short_entry[i]:
                position = -1
            if position != 0:
                exit_bar = i + params.holding_bars
            signal[i] = position
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    params = GapDebtParams()
    index = pd.date_range("2020-01-01", periods=40, freq="B")
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = GeneratedStrategy.indicators(data, params)
    assert not indicators["long_entry"].any() and not indicators["short_entry"].any()
    indicators.loc[index[1], "long_entry"] = True
    indicators.loc[index[3], "short_entry"] = True
    indicators.loc[index[12], "short_entry"] = True
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    expected = np.zeros(40, dtype=int)
    expected[2:10] = 1
    expected[13:21] = -1
    assert np.array_equal(result["signal"].to_numpy(), expected)
    assert result.index.equals(data.index) and (result["size"] > 0).all()

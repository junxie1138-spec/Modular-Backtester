from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    atr_period: int = 14
    min_gap: float = 0.001
    min_return: float = 0.0005
    stop_atr: float = 1.25
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791203543"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(params.atr_period + 1, 3)

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if (params.atr_period < 1 or params.min_gap <= 0
                or params.min_return <= 0 or params.stop_atr <= 0
                or params.hold_bars not in (1, 2)):
            raise ValueError("Positive thresholds and ATR period; hold_bars must be 1 or 2.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1).where(lambda x: x > 0)
        out["cc_return"] = data["close"] / out["previous_close"] - 1.0
        out["gap"] = data["open"] / out["previous_close"] - 1.0
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["confirmation"] = (
            (out["gap"] <= -params.min_gap)
            & (out["cc_return"] >= params.min_return)
        )
        out["entry_ready"] = (
            out["confirmation"]
            & out["confirmation"].shift(1, fill_value=False)
            & (out["gap"] < out["gap"].shift(1))
            & (out["cc_return"] > out["cc_return"].shift(1))
            & out["atr"].gt(0)
        ).fillna(False)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: StrategyParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ready = indicators["entry_ready"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        pending = False
        active = False
        held = 0
        stop = np.nan
        entry_atr = np.nan

        for i in range(len(data)):
            if pending:
                pending = False
                if np.isfinite(opens[i]) and opens[i] > 0:
                    # Entry fills at this open; freeze the previous close's ATR.
                    stop = opens[i] - params.stop_atr * entry_atr
                    active = True
                    held = 0
            if active:
                held += 1
                if (not np.isfinite(closes[i]) or closes[i] < stop
                        or held >= params.hold_bars):
                    active = False
                else:
                    decisions[i] = 1
                # Never replace an exit with a fresh entry on the same close.
                continue
            if (i >= self.warmup_bars(params) - 1 and ready[i]
                    and np.isfinite(atr[i]) and atr[i] > 0):
                decisions[i] = 1
                pending = True
                entry_atr = atr[i]

        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = StrategyParams(atr_period=1)
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({
        "open": 100.0, "high": 110.0, "low": 95.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    context = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    indicators = pd.DataFrame({
        "atr": 2.0, "entry_ready": [False, False, True, False, False, False, False],
    }, index=index)
    result = strategy.generate_signals(data, indicators, context, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    data.loc[index[3], "close"] = 97.0
    result = strategy.generate_signals(data, indicators, context, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 0]
    data.loc[index[3], "close"] = 108.0
    data.loc[index[4], "close"] = 99.0
    indicators.loc[index[3]:, "atr"] = 0.1
    data.loc[index[3], "close"] = 98.0
    result = strategy.generate_signals(data, indicators, context, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 1, 0, 0]
    params.hold_bars = 1
    result = strategy.generate_signals(data, indicators, context, params).data
    assert result["signal"].tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert result["size"].gt(0).all()
    assert strategy.indicators(data, params).index.equals(data.index)
    data["close"] = [100.0, 100.0, 100.1, 100.3, 100.3, 100.3, 100.3]
    data["open"] = [100.0, 100.0, 99.8, 99.7, 100.3, 100.3, 100.3]
    derived = strategy.indicators(data, params)
    assert not derived["entry_ready"].iloc[2]
    assert derived["entry_ready"].iloc[3]


if __name__ == "__main__":
    _self_check()

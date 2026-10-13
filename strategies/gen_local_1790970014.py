from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    profit_target: float = 0.015
    holding_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local_1790970014"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    @staticmethod
    def warmup_bars(params: StrategyParams) -> int:
        return 15

    @staticmethod
    def indicators(data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float)
        out["log_close"] = np.log(close.where(close > 0))
        out["trend_mean"] = out["log_close"].rolling(12, min_periods=12).mean()
        # Fixed-window least-squares slope, oldest observation at x=0.
        weighted = sum((5.5 - lag) * out["log_close"].shift(lag) for lag in range(12))
        out["slope"] = weighted / 143.0
        out["fitted_close"] = np.exp(out["trend_mean"] + 5.5 * out["slope"])
        out["predicted_slope"] = 2.0 * out["slope"].shift(1) - out["slope"].shift(2)
        out["slope_innovation"] = out["slope"] - out["predicted_slope"]
        out["previous_close"] = close.shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1)
        out["average_range"] = out["true_range"].rolling(12, min_periods=12).mean()
        out["discount"] = (out["fitted_close"] - close) / out["average_range"].where(out["average_range"] > 0)
        out["innovation_scale"] = out["average_range"] / close.where(close > 0)
        out["innovation_score"] = out["slope_innovation"] / out["innovation_scale"].where(out["innovation_scale"] > 0)
        out = out.replace([np.inf, -np.inf], np.nan)
        out["entry"] = (
            (out["slope"] > 0)
            & out["discount"].between(0.2, 1.5)
            & (out["innovation_score"] > 0.03)
            & (out["innovation_score"].shift(1) <= 0.03)
            & (close > out["previous_close"])
        ).fillna(False)
        return out

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        if params.holding_bars < 1 or not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("holding_bars must be positive and profit_target must be finite and positive")
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        active = False
        entry_price = np.nan
        held = 0
        for i in range(len(data)):
            if active:
                # The preceding close's entry decision fills at this bar's open.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                if not np.isfinite(entry_price) or entry_price <= 0:
                    active = False
                    entry_price = np.nan
                    held = 0
                    continue
                held += 1
                target_hit = np.isfinite(closes[i]) and closes[i] >= entry_price * (1.0 + params.profit_target)
                if target_hit or held >= params.holding_bars:
                    active = False
                    entry_price = np.nan
                    held = 0
                else:
                    signals[i] = 1
            elif entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                active = True
                entry_price = np.nan
                held = 0
                signals[i] = 1
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entry": [True, False, False, False, False, False, False, False]}, index=index)
    params = StrategyParams()
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    data.loc[index[2], "close"] = 102.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 0, 0, 0, 0, 0]
    data.loc[index[1], "open"] = 103.0
    result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()

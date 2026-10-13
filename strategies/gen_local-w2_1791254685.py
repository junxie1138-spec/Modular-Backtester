from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    ma_window: int = 20
    session_window: int = 5
    atr_window: int = 14
    z_threshold: float = 1.0
    session_dominance: float = 1.25
    stop_atr: float = 1.5
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791254685"

    @classmethod
    def params_type(cls):
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return int(max(params.ma_window + 1, params.session_window + 1, params.atr_window + 1))

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        if min(params.ma_window, params.session_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two bars")
        if not all(np.isfinite(x) and x > 0 for x in (params.z_threshold, params.session_dominance, params.stop_atr)):
            raise ValueError("Thresholds and stop multiplier must be finite and positive")
        if params.hold_bars < 1:
            raise ValueError("hold_bars must be positive")
        close = data["close"]
        previous = close.shift(1)
        mean = close.rolling(params.ma_window, min_periods=params.ma_window).mean()
        deviation = close.rolling(params.ma_window, min_periods=params.ma_window).std(ddof=0)
        z = (close - mean) / deviation.replace(0.0, np.nan)
        session_return = np.log(close.where(close > 0) / data["open"].where(data["open"] > 0))
        gap_return = np.log(data["open"].where(data["open"] > 0) / previous.where(previous > 0))
        session_displacement = session_return.rolling(params.session_window, min_periods=params.session_window).sum()
        gap_budget = gap_return.abs().rolling(params.session_window, min_periods=params.session_window).sum()
        dominance = session_displacement / gap_budget.clip(lower=1e-8)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1).where(previous.notna())
        atr = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        long_entry = (z >= params.z_threshold) & (z.shift(1) < params.z_threshold) & (dominance >= params.session_dominance)
        short_entry = (z <= -params.z_threshold) & (z.shift(1) > -params.z_threshold) & (dominance <= -params.session_dominance)
        direction = np.select([long_entry.fillna(False), short_entry.fillna(False)], [1, -1], default=0)
        return pd.DataFrame({
            "ma": mean,
            "price_std": deviation,
            "z": z,
            "session_return": session_return,
            "gap_return": gap_return,
            "session_displacement": session_displacement,
            "gap_budget": gap_budget,
            "session_dominance": dominance,
            "true_range": true_range,
            "atr": atr,
            "entry_direction": direction,
        }, index=data.index).replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        decisions = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        position = 0
        entry_price = np.nan
        stop_distance = np.nan
        age = 0
        for i in range(len(data)):
            if position:
                # The preceding close requested entry; this bar's open is its fill reference.
                if np.isnan(entry_price):
                    entry_price = opens[i]
                age += 1
                valid = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                stopped = valid and position * (closes[i] - entry_price) < -stop_distance
                if not valid or stopped or age >= params.hold_bars:
                    position = 0
                    entry_price = np.nan
                    age = 0
            elif entries[i] in (-1, 1) and np.isfinite(atr[i]) and atr[i] > 0 and np.isfinite(closes[i]):
                position = int(entries[i])
                entry_price = np.nan
                stop_distance = params.stop_atr * atr[i]
                age = 0
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = StrategyParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": [100, 110, 100, 100, 90, 100, 100], "close": [100, 106, 100, 100, 94, 100, 100]}, index=index)
    indicators = pd.DataFrame({"atr": 2.0, "entry_direction": [1, 0, 0, -1, 0, 0, 0]}, index=index)
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    data.loc[:, "open"] = 100.0
    data.loc[:, "close"] = 100.0
    indicators["entry_direction"] = [1, 0, 0, 0, 0, 0, 0]
    result = GeneratedStrategy.generate_signals(None, data, indicators, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0, 0]
    assert result["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()

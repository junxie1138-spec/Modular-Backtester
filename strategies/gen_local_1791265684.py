from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeAcceptanceParams:
    range_bars: int = 40
    streak_bars: int = 5
    min_position: float = 0.60
    max_position: float = 0.90
    atr_bars: int = 14
    breakeven_gain: float = 0.025
    trail_atr: float = 2.5
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[RangeAcceptanceParams]):
    strategy_id = "gen_local_1791265684"

    @classmethod
    def params_type(cls):
        return RangeAcceptanceParams

    @staticmethod
    def warmup_bars(params: RangeAcceptanceParams) -> int:
        return max(200, params.range_bars + params.streak_bars, params.atr_bars + 1)

    @staticmethod
    def indicators(data: pd.DataFrame, params: RangeAcceptanceParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        out["ma200"] = data["close"].rolling(200, min_periods=200).mean()
        out["range_high"] = data["high"].rolling(params.range_bars, min_periods=params.range_bars).max().shift(1)
        out["range_low"] = data["low"].rolling(params.range_bars, min_periods=params.range_bars).min().shift(1)
        width = (out["range_high"] - out["range_low"]).replace(0.0, np.nan)
        out["position"] = (data["close"] - out["range_low"]) / width
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        upper = (out["position"] >= params.min_position).fillna(False)
        out["upper_streak"] = upper.astype(int).groupby((~upper).cumsum()).cumsum()
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RangeAcceptanceParams,
    ) -> SignalFrame:
        entry_condition = (
            (indicators["upper_streak"] == params.streak_bars)
            & (indicators["position"] <= params.max_position)
            & (data["close"] > indicators["ma200"])
            & (indicators["atr"] > 0.0)
        ).fillna(False).to_numpy(dtype=bool)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        regime = indicators["ma200"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry = 0.0
        stop = 0.0
        armed = False
        age = 0
        for i in range(len(data)):
            if active:
                # The previous close's entry decision fills at this bar's open.
                if entry == 0.0:
                    if i == 0 or not np.isfinite(opens[i]) or opens[i] <= 0.0 or not np.isfinite(atr[i - 1]):
                        active = False
                        continue
                    entry = opens[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                age += 1
                valid = np.isfinite(closes[i]) and np.isfinite(atr[i]) and np.isfinite(regime[i])
                if not valid or closes[i] <= stop or closes[i] < regime[i] or age >= params.hold_bars:
                    active = False
                else:
                    # Close-confirmed stops produce exits at the following open.
                    armed = armed or closes[i] >= entry * (1.0 + params.breakeven_gain)
                    if armed:
                        stop = max(stop, entry, closes[i] - params.trail_atr * atr[i])
            elif entry_condition[i]:
                active = True
                entry = 0.0
                armed = False
                age = 0
            signal[i] = int(active)
        df = pd.DataFrame({"signal": signal, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = RangeAcceptanceParams()
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame({
        "open": 100.0, "high": 104.0, "low": 99.0,
        "close": 100.0, "volume": 1000.0,
    }, index=index)
    ind = pd.DataFrame({
        "ma200": 90.0, "position": 0.75, "atr": 1.0,
        "upper_streak": 0,
    }, index=index)
    ind.iloc[0, ind.columns.get_loc("upper_streak")] = params.streak_bars
    data.iloc[1, data.columns.get_loc("close")] = 103.0
    data.iloc[2, data.columns.get_loc("close")] = 102.0
    ind.iloc[2, ind.columns.get_loc("atr")] = 5.0
    data.iloc[3, data.columns.get_loc("close")] = 100.4
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[:5].tolist() == [0, 1, 1, 1, 0]
    data["close"] = 100.0
    ind["atr"] = 1.0
    result = GeneratedStrategy.generate_signals(data, ind, None, params).data
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19:].eq(0).all()
    assert result["size"].gt(0.0).all()
    ind["ma200"] = np.nan
    assert GeneratedStrategy.generate_signals(data, ind, None, params).data["signal"].eq(0).all()
    assert GeneratedStrategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()

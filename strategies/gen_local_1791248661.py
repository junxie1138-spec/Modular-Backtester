from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class AbsorptionParams:
    streak_bars: int = 3
    atr_bars: int = 14
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[AbsorptionParams]):
    strategy_id = "gen_local_1791248661"

    @classmethod
    def params_type(cls) -> type[AbsorptionParams]:
        return AbsorptionParams

    def warmup_bars(self, params: AbsorptionParams) -> int:
        return max(params.atr_bars + 1, params.streak_bars + 1)

    def indicators(self, data: pd.DataFrame, params: AbsorptionParams) -> pd.DataFrame:
        if params.streak_bars < 1 or params.atr_bars < 1 or params.max_hold < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not np.isfinite(params.trail_atr) or params.trail_atr <= 0:
            raise ValueError("trail_atr must be finite and positive")
        close = data["close"]
        previous = close.shift(1)
        change = close.diff()
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1)
        true_range = true_range.where(previous.notna())
        out = pd.DataFrame(index=data.index)
        out["atr"] = true_range.rolling(params.atr_bars, min_periods=params.atr_bars).mean()
        valid = np.isfinite(data[["open", "high", "low", "close"]]).all(axis=1)
        valid &= previous.notna() & (close > 0)
        conditions = {
            "bull": valid & (change > 0) & (close < data["open"]),
            "bear": valid & (change < 0) & (close > data["open"]),
        }
        for name, condition in conditions.items():
            groups = (~condition).cumsum()
            out[name + "_streak"] = condition.astype(int).groupby(groups).cumsum()
            out[name + "_retained"] = change.abs().where(condition, 0.0).groupby(groups).cumsum()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: AbsorptionParams,
    ) -> SignalFrame:
        df = pd.DataFrame({"signal": 0, "size": 0.25}, index=data.index)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.full(len(data), 0.25, dtype=float)
        direction = 0
        age = 0
        extreme = np.nan
        stop = np.nan
        conviction = 0.25
        closes = data["close"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        bull = indicators["bull_streak"].fillna(0).to_numpy()
        bear = indicators["bear_streak"].fillna(0).to_numpy()
        bull_move = indicators["bull_retained"].fillna(0).to_numpy(dtype=float)
        bear_move = indicators["bear_retained"].fillna(0).to_numpy(dtype=float)
        for i in range(len(data)):
            close, atr = closes[i], atrs[i]
            if not np.isfinite(close) or not np.isfinite(atr) or close <= 0 or atr <= 0:
                direction = 0
                age = 0
                extreme = stop = np.nan
                continue
            if direction:
                age += 1
                if age == 1:
                    # Entry fills at this bar's open; initialize the trail after entry.
                    reference = opens[i] if np.isfinite(opens[i]) else close
                    extreme = reference
                    stop = reference - direction * params.trail_atr * atr
                if direction == 1:
                    extreme = max(extreme, close)
                    stop = max(stop, extreme - params.trail_atr * atr)
                    stopped = close <= stop
                else:
                    extreme = min(extreme, close)
                    stop = min(stop, extreme + params.trail_atr * atr)
                    stopped = close >= stop
                if stopped or age >= params.max_hold:
                    direction = 0
                    age = 0
                    extreme = stop = np.nan
                signals[i] = direction
                sizes[i] = conviction
                continue
            previous_bull = bull[i - 1] if i else 0
            previous_bear = bear[i - 1] if i else 0
            if bull[i] >= params.streak_bars and previous_bull < params.streak_bars:
                direction, count, retained = 1, bull[i], bull_move[i]
            elif bear[i] >= params.streak_bars and previous_bear < params.streak_bars:
                direction, count, retained = -1, bear[i], bear_move[i]
            else:
                continue
            strength = count / (params.streak_bars + 2.0) + retained / (4.0 * atr)
            conviction = float(np.clip(0.25 + 0.35 * strength, 0.25, 0.95))
            age = 0
            extreme = stop = np.nan
            signals[i] = direction
            sizes[i] = conviction
        df["signal"] = signals
        df["size"] = sizes
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(0.25).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = AbsorptionParams(streak_bars=2, atr_bars=2, trail_atr=2.0, max_hold=8)
    index = pd.date_range("2020-01-01", periods=8)
    for direction in (1, -1):
        closes = np.array([100, 101, 102, 104, 103, 101, 100, 99], dtype=float)
        if direction == -1:
            closes = 200.0 - closes
        data = pd.DataFrame({"open": closes, "high": closes + 1,
                             "low": closes - 1, "close": closes, "volume": 1000}, index=index)
        ind = pd.DataFrame({"atr": [1, 1, 1, 1, 10, 10, 10, 10],
                            "bull_streak": 0, "bear_streak": 0,
                            "bull_retained": 0.0, "bear_retained": 0.0}, index=index)
        side = "bull" if direction == 1 else "bear"
        ind.loc[index[1], side + "_streak"] = 2
        ind.loc[index[1], side + "_retained"] = 1.0
        result = GeneratedStrategy.generate_signals(None, data, ind, None, params).data
        assert result["signal"].tolist() == [0, 0, direction, direction, direction, direction, 0, 0]
        assert result["size"].between(0.25, 0.95).all()
        stronger = ind.copy()
        stronger.loc[index[1], side + "_retained"] = 3.0
        scaled = GeneratedStrategy.generate_signals(None, data, stronger, None, params).data
        assert scaled["size"].iloc[2] > result["size"].iloc[2]


if __name__ == "__main__":
    _self_check()

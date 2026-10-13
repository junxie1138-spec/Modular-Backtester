from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RegimeStreakParams:
    confirmation_days: int = 9
    atr_days: int = 14
    holding_bars: int = 18
    trail_atr: float = 3.5
    max_distance_atr: float = 3.0


class GeneratedStrategy(BaseStrategy[RegimeStreakParams]):
    strategy_id = "gen_local-w2_1791259224"

    @classmethod
    def params_type(cls) -> type[RegimeStreakParams]:
        return RegimeStreakParams

    @staticmethod
    def warmup_bars(params: RegimeStreakParams) -> int:
        if min(params.confirmation_days, params.atr_days, params.holding_bars) < 1:
            raise ValueError("Lookbacks and holding period must be positive")
        if not all(np.isfinite(x) and x > 0 for x in (params.trail_atr, params.max_distance_atr)):
            raise ValueError("ATR multipliers must be finite and positive")
        return int(max(200 + params.confirmation_days - 1, params.atr_days + 1))

    @staticmethod
    def indicators(data: pd.DataFrame, params: RegimeStreakParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        previous = close.shift(1)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous).abs(),
            (data["low"] - previous).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_days, min_periods=params.atr_days).mean()
        out["above"] = close.gt(out["ma200"])
        out["below"] = close.lt(out["ma200"])
        for side in ("above", "below"):
            active = out[side]
            out[side + "_streak"] = active.astype(int).groupby((~active).cumsum()).cumsum()
        out["distance_atr"] = (close - out["ma200"]).abs().div(out["atr"].where(out["atr"] > 0))
        out["valid"] = (
            np.isfinite(close) & np.isfinite(out["ma200"])
            & np.isfinite(out["atr"]) & out["atr"].gt(0)
            & np.isfinite(out["distance_atr"])
        )
        eligible = out["valid"] & out["distance_atr"].le(params.max_distance_atr)
        out["long_entry"] = eligible & out["above_streak"].eq(params.confirmation_days)
        out["short_entry"] = eligible & out["below_streak"].eq(params.confirmation_days)
        return out

    @staticmethod
    def generate_signals(
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: RegimeStreakParams,
    ) -> SignalFrame:
        GeneratedStrategy.warmup_bars(params)
        closes = data["close"].to_numpy(dtype=float)
        atrs = indicators["atr"].to_numpy(dtype=float)
        averages = indicators["ma200"].to_numpy(dtype=float)
        valid = indicators["valid"].fillna(False).to_numpy(dtype=bool)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        signals = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        highest_close = lowest_close = stop = 0.0
        for i in range(len(data)):
            if not valid[i]:
                position = 0
                continue
            close, atr = closes[i], atrs[i]
            if position:
                highest_close = max(highest_close, close)
                lowest_close = min(lowest_close, close)
                if position == 1:
                    stop = max(stop, highest_close - params.trail_atr * atr)
                    stopped = close <= stop
                    regime_lost = close <= averages[i]
                else:
                    # Mirror the long high-water trail for short exposure.
                    stop = min(stop, lowest_close + params.trail_atr * atr)
                    stopped = close >= stop
                    regime_lost = close >= averages[i]
                if stopped or regime_lost or i - entry_bar >= params.holding_bars:
                    position = 0
                signals[i] = position
                continue
            if longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = i
                highest_close = lowest_close = close
                stop = close - position * params.trail_atr * atr
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"close": [100., 110., 105., 100., 90., 95., 100.]}, index=index)
    derived = pd.DataFrame({
        "atr": 1.0,
        "ma200": [99., 99., 99., 101., 101., 101., 101.],
        "valid": True,
        "long_entry": [True, False, False, False, False, False, False],
        "short_entry": [False, False, False, True, False, False, False],
    }, index=index)
    result = GeneratedStrategy.generate_signals(data, derived, None, RegimeStreakParams())
    assert result.data["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    assert result.data["size"].gt(0).all()

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StreakParams:
    streak_length: int = 3
    stop_atr: float = 2.0


class GeneratedStrategy(BaseStrategy[StreakParams]):
    strategy_id = "gen_local-w2_1791286672"

    @classmethod
    def params_type(cls) -> type[StreakParams]:
        return StreakParams

    def warmup_bars(self, params: StreakParams) -> int:
        return max(129, params.streak_length + 1)

    def indicators(self, data: pd.DataFrame, params: StreakParams) -> pd.DataFrame:
        if params.streak_length < 2 or not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("streak_length must be >= 2 and stop_atr must be positive")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        direction = np.sign(close.diff()).fillna(0).astype(int)
        groups = direction.ne(direction.shift()).cumsum()
        streak = (direction.groupby(groups).cumcount() + 1).where(direction.ne(0), 0)
        previous_direction = direction.shift(1)
        mature = streak.shift(1).ge(2)
        continued = direction.eq(previous_direction)
        out["direction"] = direction
        out["streak"] = streak
        for side, name in ((1, "up_survival"), (-1, "down_survival")):
            eligible = mature & previous_direction.eq(side)
            trials = eligible.astype(float).rolling(126, min_periods=126).sum()
            successes = (eligible & continued).astype(float).rolling(126, min_periods=126).sum()
            out[name] = successes.div(trials.where(trials.ge(6)))
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = true_range.rolling(14, min_periods=14).mean()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: StreakParams,
    ) -> SignalFrame:
        direction = indicators["direction"].fillna(0).to_numpy(dtype=int)
        streak = indicators["streak"].fillna(0).to_numpy()
        survival = indicators["up_survival"].where(
            indicators["direction"].eq(1), indicators["down_survival"]
        ).to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        signals = np.zeros(len(data), dtype=int)
        side = 0
        pending_entry = False
        stop_price = np.nan
        entry_atr = np.nan
        held = 0
        for i in range(len(data)):
            if side:
                if pending_entry:
                    if not np.isfinite(opens[i]) or opens[i] <= 0:
                        side = 0
                        pending_entry = False
                        continue
                    stop_price = opens[i] - side * params.stop_atr * entry_atr
                    pending_entry = False
                held += 1
                stop_hit = np.isfinite(closes[i]) and side * (closes[i] - stop_price) < 0
                if stop_hit or held >= 8:
                    side = 0
            elif (
                direction[i] != 0
                and streak[i] == params.streak_length
                and np.isfinite(survival[i])
                and survival[i] >= 0.60
                and np.isfinite(atr[i])
                and atr[i] > 0
                and np.isfinite(closes[i])
                and closes[i] > 0
            ):
                side = int(direction[i])
                entry_atr = float(atr[i])
                pending_entry = True
                held = 0
            signals[i] = side
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    index = pd.date_range("2020-01-01", periods=12)
    for side in (1, -1):
        data = pd.DataFrame({"open": 101.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000.0}, index=index)
        features = pd.DataFrame({"direction": side, "streak": 0, "up_survival": 0.7, "down_survival": 0.7, "atr": 1.0}, index=index)
        features.loc[index[0], "streak"] = 3
        result = GeneratedStrategy.generate_signals(None, data, features, None, StreakParams()).data
        assert result["signal"].tolist() == [0] + [side] * 8 + [0] * 3
        data.loc[index[2], "close"] = 101.0 - side * 3.0
        result = GeneratedStrategy.generate_signals(None, data, features, None, StreakParams()).data
        assert result["signal"].tolist() == [0, side, side] + [0] * 9
        assert result["size"].gt(0).all()

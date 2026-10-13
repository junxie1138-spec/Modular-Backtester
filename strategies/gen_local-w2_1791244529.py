from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GapRepairParams:
    gap_window: int = 20
    min_streak: int = 2
    streak_cap: int = 6
    gap_multiple: float = 0.75
    profit_target: float = 0.006
    max_hold: int = 2
    size_floor: float = 0.25
    size_cap: float = 0.95


class GeneratedStrategy(BaseStrategy[GapRepairParams]):
    strategy_id = "gen_local-w2_1791244529"

    @classmethod
    def params_type(cls) -> type[GapRepairParams]:
        return GapRepairParams

    @staticmethod
    def warmup_bars(params: GapRepairParams) -> int:
        return int(max(params.gap_window + 2, params.streak_cap + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: GapRepairParams) -> pd.DataFrame:
        if params.gap_window < 2 or not 1 <= params.min_streak <= params.streak_cap:
            raise ValueError("Invalid gap window or streak bounds")
        if params.gap_multiple <= 0 or params.profit_target <= 0:
            raise ValueError("Gap multiple and profit target must be positive")
        if params.max_hold not in (1, 2) or not 0 < params.size_floor <= params.size_cap <= 1:
            raise ValueError("Invalid holding period or size bounds")
        previous = data["close"].shift(1).where(lambda x: x > 0)
        opening = data["open"].where(lambda x: x > 0)
        gap = (opening / previous - 1).replace([np.inf, -np.inf], np.nan)
        intraday = (data["close"] / opening - 1).replace([np.inf, -np.inf], np.nan)
        scale = gap.abs().rolling(params.gap_window).median().shift(1).replace(0, np.nan)
        repaired = pd.Series(np.select([
            (gap > 0) & (data["close"] <= previous),
            (gap < 0) & (data["close"] >= previous),
        ], [1, -1], default=0), index=data.index, dtype=int)
        groups = repaired.ne(repaired.shift(1)).cumsum()
        streak = (repaired.groupby(groups).cumcount() + 1).clip(upper=params.streak_cap) * repaired
        prior = streak.shift(1).fillna(0)
        valid = (
            scale.notna() & gap.notna() & intraday.notna()
            & (prior.abs() >= params.min_streak)
            & (np.sign(gap) == np.sign(prior))
            & (np.sign(intraday) == np.sign(prior))
            & (gap.abs() >= params.gap_multiple * scale)
        )
        candidate = pd.Series(np.where(valid, np.sign(prior), 0), index=data.index, dtype=int)
        strength = (
            0.5 * (prior.abs() / params.streak_cap).clip(0, 1)
            + 0.5 * (intraday.abs() / scale).clip(0, 1)
        ).fillna(0).clip(0, 1)
        entry_size = params.size_floor + (params.size_cap - params.size_floor) * strength
        return pd.DataFrame({
            "gap": gap.fillna(0),
            "intraday": intraday.fillna(0),
            "gap_scale": scale.fillna(0),
            "repair_direction": repaired,
            "repair_streak": streak,
            "prior_streak": prior,
            "candidate": candidate,
            "strength": strength,
            "entry_size": entry_size,
        }, index=data.index)

    @staticmethod
    def generate_signals(data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: GapRepairParams) -> SignalFrame:
        count = len(data)
        signals = np.zeros(count, dtype=int)
        sizes = np.full(count, params.size_floor, dtype=float)
        candidates = indicators["candidate"].to_numpy(dtype=int)
        entry_sizes = indicators["entry_size"].to_numpy(dtype=float)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        direction = 0
        entry_bar = -1
        entry_price = 0.0
        held_size = params.size_floor
        for i in range(count):
            if direction:
                # Yesterday's decision becomes a position at today's open.
                if entry_bar < 0:
                    entry_bar = i
                    entry_price = opens[i]
                valid_prices = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                gain = direction * (closes[i] / entry_price - 1) if valid_prices else 0.0
                # Targets are evaluated at the close; exits fill next open.
                if (not valid_prices or gain >= params.profit_target
                        or i - entry_bar + 1 >= params.max_hold):
                    direction = 0
                    entry_bar = -1
                    continue
                signals[i] = direction
                sizes[i] = held_size
            elif candidates[i]:
                direction = int(candidates[i])
                entry_bar = -1
                held_size = float(entry_sizes[i])
                signals[i] = direction
                sizes[i] = held_size
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(params.size_floor).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    index = pd.date_range("2020-01-01", periods=8)
    for direction, target, expected in [
        (1, 0.1, [0, 0, 0, 0, 0, 1, 1, 0]),
        (1, 0.005, [0, 0, 0, 0, 0, 1, 0, 0]),
        (-1, 0.005, [0, 0, 0, 0, 0, -1, 0, 0]),
    ]:
        opening = 100 + direction * (np.array([100, 101, 101, 101, 102, 104, 104, 104]) - 100)
        closing = 100 + direction * (np.array([100, 100, 100, 100, 103, 105, 104, 104]) - 100)
        data = pd.DataFrame({
            "open": opening,
            "high": np.maximum(opening, closing) + 1,
            "low": np.minimum(opening, closing) - 1,
            "close": closing,
            "volume": 1000000,
        }, index=index)
        params = GapRepairParams(gap_window=2, profit_target=target)
        indicators = GeneratedStrategy.indicators(data, params)
        result = GeneratedStrategy.generate_signals(data, indicators, None, params).data
        assert result["signal"].tolist() == expected
        assert result["size"].between(params.size_floor, params.size_cap).all()
        assert result["size"].iloc[5] > params.size_floor
        assert indicators.notna().all().all()


if __name__ == "__main__":
    _self_check()

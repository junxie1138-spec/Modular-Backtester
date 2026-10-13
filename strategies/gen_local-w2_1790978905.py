from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ShockwaveParams:
    atr_period: int = 14
    streak_bars: int = 2
    stretch_atr: float = 1.2
    stop_atr: float = 2.0
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[ShockwaveParams]):
    strategy_id = "gen_local-w2_1790978905"

    @classmethod
    def params_type(cls) -> type[ShockwaveParams]:
        return ShockwaveParams

    @staticmethod
    def warmup_bars(params: ShockwaveParams) -> int:
        if params.atr_period < 2 or params.streak_bars < 1:
            raise ValueError("Lookbacks must be positive; ATR needs at least two bars.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between three and five.")
        if not np.isfinite(params.stretch_atr) or params.stretch_atr <= 0:
            raise ValueError("stretch_atr must be finite and positive.")
        if not np.isfinite(params.stop_atr) or params.stop_atr <= 0:
            raise ValueError("stop_atr must be finite and positive.")
        return int(max(params.atr_period + 1, params.streak_bars + 2))

    @staticmethod
    def indicators(data: pd.DataFrame, params: ShockwaveParams) -> pd.DataFrame:
        GeneratedStrategy.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        change = data["close"].diff()
        direction = np.sign(change).fillna(0)
        groups = direction.ne(direction.shift()).cumsum()
        directional_streak = direction.groupby(groups).cumcount() + 1
        braking = (
            direction.ne(0)
            & direction.eq(direction.shift())
            & change.abs().lt(change.abs().shift())
            & data["volume"].gt(data["volume"].shift())
            & data["volume"].gt(0)
        )
        brake_groups = (~braking).cumsum()
        braking_streak = braking.astype(int).groupby(brake_groups).cumsum()
        previous_close = data["close"].shift()
        true_range = pd.concat(
            [
                data["high"] - data["low"],
                (data["high"] - previous_close).abs(),
                (data["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["change"] = change
        out["direction"] = direction
        out["directional_streak"] = directional_streak.where(direction.ne(0), 0)
        out["braking_streak"] = braking_streak
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(
            params.atr_period, min_periods=params.atr_period
        ).mean()
        out["displacement"] = data["close"].diff(params.streak_bars + 1).abs()
        out["stretch"] = out["displacement"].div(out["atr"].where(out["atr"].gt(0)))
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: ShockwaveParams,
    ) -> SignalFrame:
        self.warmup_bars(params)
        ready = (
            indicators["atr"].gt(0)
            & indicators["stretch"].ge(params.stretch_atr)
            & indicators["directional_streak"].ge(params.streak_bars + 1)
            & indicators["braking_streak"].ge(params.streak_bars)
        ).fillna(False)
        setups = (-indicators["direction"]).where(ready, 0).fillna(0).astype(int)
        conviction = (
            0.35
            + 0.30 * (indicators["stretch"] / params.stretch_atr - 1).clip(0, 1)
            + 0.30 * (
                indicators["braking_streak"] / params.streak_bars - 1
            ).clip(0, 1)
        ).clip(0.35, 0.95).fillna(0.35)
        signals = np.zeros(len(data), dtype=int)
        sizes = np.ones(len(data), dtype=float)
        side = 0
        entry_bar = -1
        entry_price = np.nan
        entry_atr = np.nan
        active_size = 1.0
        for i in range(len(data)):
            close = float(data["close"].iloc[i])
            if side:
                # The prior close's decision fills at this bar's open.
                if i == entry_bar:
                    entry_price = float(data["open"].iloc[i])
                valid = np.isfinite(close) and np.isfinite(entry_price) and entry_price > 0
                stopped = valid and side * (close - entry_price) < -params.stop_atr * entry_atr
                expired = i - entry_bar + 1 >= params.hold_bars
                if not valid or stopped or expired:
                    side = 0
            elif ready.iloc[i] and np.isfinite(close) and close > 0:
                side = int(setups.iloc[i])
                entry_bar = i + 1
                entry_price = np.nan
                entry_atr = float(indicators["atr"].iloc[i])
                active_size = float(conviction.iloc[i])
            signals[i] = side
            sizes[i] = active_size if side else 1.0
        df = pd.DataFrame({"signal": signals, "size": sizes}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = df["size"].shift(1).fillna(1.0).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    params = ShockwaveParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        index=index,
    )
    indicators = pd.DataFrame(
        {"atr": 1.0, "stretch": 0.0, "direction": 0.0,
         "directional_streak": 0, "braking_streak": 0},
        index=index,
    )
    strategy = GeneratedStrategy()
    for side in (1, -1):
        indicators.iloc[0] = [1.0, 2.4, -side, 3, 2]
        data["close"] = 100.0
        held = strategy.generate_signals(data, indicators, None, params).data
        assert held["signal"].tolist() == [0, side, side, side, side, 0, 0]
        data.loc[index[2], "close"] = 100.0 - side * 3.0
        stopped = strategy.generate_signals(data, indicators, None, params).data
        assert stopped["signal"].tolist() == [0, side, side, 0, 0, 0, 0]
        assert stopped["size"].gt(0).all()
        data["close"] = 100.0
        indicators.loc[index[0], "stretch"] = params.stretch_atr
        weaker = strategy.generate_signals(data, indicators, None, params).data
        assert weaker["size"].iloc[1] < held["size"].iloc[1]
    actual = strategy.indicators(data, params)
    assert actual.index.equals(data.index)
    assert strategy.generate_signals(data, actual, None, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()

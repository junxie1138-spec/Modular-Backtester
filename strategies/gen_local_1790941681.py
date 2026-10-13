from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    peak_window: int = 20
    loss_streak: int = 3
    min_drawdown: float = 0.025
    hold_bars: int = 2


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790941681"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if params.peak_window < 2 or params.loss_streak < 2:
            raise ValueError("Lookbacks must be at least two bars.")
        if not np.isfinite(params.min_drawdown) or not 0 < params.min_drawdown < 1:
            raise ValueError("min_drawdown must lie between zero and one.")
        if params.hold_bars not in (1, 2):
            raise ValueError("hold_bars must be one or two.")
        return int(max(params.peak_window, params.loss_streak + 1) + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(np.isfinite(data["close"]) & data["close"].gt(0))
        out["change"] = close.diff()
        down = out["change"].lt(0)
        # Saturation makes the streak depend on at most loss_streak changes.
        out["loss_count"] = down.groupby((~down).cumsum()).cumsum().clip(upper=params.loss_streak)
        out["peak"] = data["high"].rolling(params.peak_window, min_periods=params.peak_window).max()
        out["drawdown"] = (1.0 - close / out["peak"].where(out["peak"].gt(0))).replace([np.inf, -np.inf], np.nan)
        out["selloff_midpoint"] = ((data["high"] + data["low"]) / 2.0).shift(2)
        out["setup"] = (
            out["loss_count"].shift(2).ge(params.loss_streak)
            & out["drawdown"].shift(2).ge(params.min_drawdown)
        )
        out["two_rising_closes"] = out["change"].gt(0) & out["change"].shift(1).gt(0)
        midpoint = out["selloff_midpoint"]
        out["reclaimed_twice"] = close.shift(1).gt(midpoint) & close.gt(midpoint)
        out["failed_twice"] = close.shift(1).lt(midpoint) & close.lt(midpoint)
        ready = out["setup"] & out["two_rising_closes"]
        out["entry"] = np.select(
            [ready & out["reclaimed_twice"], ready & out["failed_twice"]],
            [1, -1],
            default=0,
        ).astype(int)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        entries = indicators["entry"].fillna(0).to_numpy(dtype=int)
        signal = np.zeros(len(data), dtype=int)
        position = 0
        exit_at = -1
        for i in range(len(data)):
            if position:
                if i == exit_at:
                    position = 0
                    # Preserve a flat exit bar even if another setup appears.
                else:
                    signal[i] = position
            elif i >= warmup - 1 and entries[i] != 0:
                position = int(entries[i])
                exit_at = i + params.hold_bars
                signal[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = signal
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 1.0
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    prices = [100.0] * 24 + [99.0, 98.0, 96.0, 96.8, 97.1, 97.2, 97.3, 97.4]
    close = pd.Series(prices, index=pd.date_range("2020-01-01", periods=len(prices)))
    data = pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": 1000.0})
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    for selloff_high, direction in ((96.2, 1), (100.0, -1)):
        data.iloc[26, data.columns.get_loc("high")] = selloff_high
        for hold in (1, 2):
            params = GeneratedParams(hold_bars=hold)
            indicators = strategy.indicators(data, params)
            signals = strategy.generate_signals(data, indicators, ctx, params).data
            expected = [0] * len(data)
            expected[29:29 + hold] = [direction] * hold
            assert signals["signal"].tolist() == expected
            assert signals["size"].gt(0).all()
        prefix = strategy.indicators(data.iloc[:29], params)
        pd.testing.assert_frame_equal(prefix, indicators.iloc[:29])

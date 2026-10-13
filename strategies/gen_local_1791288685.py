from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class VisibilityParams:
    window: int = 8
    hub_min: int = 4
    hub_drop: int = 2
    profit_target: float = 0.02
    max_hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[VisibilityParams]):
    strategy_id = "gen_local_1791288685"

    @classmethod
    def params_type(cls):
        return VisibilityParams

    def warmup_bars(self, params: VisibilityParams) -> int:
        if not 3 <= params.window <= 9:
            raise ValueError("window must be between 3 and 9")
        if params.hub_min < 1 or params.hub_drop < 1:
            raise ValueError("hub thresholds must be positive")
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be finite and positive")
        if params.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be positive")
        return int(params.window + 1)

    def indicators(self, data: pd.DataFrame, params: VisibilityParams) -> pd.DataFrame:
        self.warmup_bars(params)
        close = data["close"].astype(float).where(lambda x: np.isfinite(x) & (x > 0))
        upper_degree = pd.Series(0.0, index=data.index)
        lower_degree = pd.Series(0.0, index=data.index)
        # Natural visibility: every intermediate close lies below the chord.
        # Reversing the inequality computes visibility on inverted prices.
        for lag in range(1, params.window):
            past = close.shift(lag)
            upper_visible = close.notna() & past.notna()
            lower_visible = upper_visible.copy()
            for offset in range(1, lag):
                intermediate = close.shift(offset)
                chord = close + (past - close) * (offset / lag)
                upper_visible &= intermediate.lt(chord)
                lower_visible &= intermediate.gt(chord)
            upper_degree += upper_visible.astype(float)
            lower_degree += lower_visible.astype(float)
        ready = close.rolling(params.window).count().eq(params.window)
        out = pd.DataFrame(index=data.index)
        out["upper_degree"] = upper_degree.where(ready)
        out["lower_degree"] = lower_degree.where(ready)
        out["previous_upper_degree"] = out["upper_degree"].shift(1)
        out["previous_lower_degree"] = out["lower_degree"].shift(1)
        out["local_mean"] = close.rolling(params.window, min_periods=params.window).mean()
        out["close_change"] = close.diff()
        out["long_entry"] = (
            out["previous_lower_degree"].ge(params.hub_min)
            & (out["previous_lower_degree"] - out["lower_degree"]).ge(params.hub_drop)
            & out["previous_lower_degree"].gt(out["previous_upper_degree"])
            & close.lt(out["local_mean"])
            & out["close_change"].gt(0)
        ).fillna(False)
        out["short_entry"] = (
            out["previous_upper_degree"].ge(params.hub_min)
            & (out["previous_upper_degree"] - out["upper_degree"]).ge(params.hub_drop)
            & out["previous_upper_degree"].gt(out["previous_lower_degree"])
            & close.gt(out["local_mean"])
            & out["close_change"].lt(0)
        ).fillna(False)
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: VisibilityParams,
    ) -> SignalFrame:
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        entry_price = np.nan
        for i in range(len(data)):
            if position:
                # An entry decided at the previous close fills at this open.
                if i == entry_bar:
                    entry_price = opens[i]
                held_bars = i - entry_bar + 1
                valid_price = (
                    np.isfinite(entry_price) and entry_price > 0
                    and np.isfinite(closes[i]) and closes[i] > 0
                )
                target_hit = (
                    valid_price
                    and position * (closes[i] / entry_price - 1.0) >= params.profit_target
                )
                if target_hit or held_bars >= params.max_hold_bars:
                    position = 0
                    entry_price = np.nan
            elif np.isfinite(closes[i]) and closes[i] > 0:
                if longs[i] and not shorts[i]:
                    position = 1
                elif shorts[i] and not longs[i]:
                    position = -1
                if position:
                    entry_bar = i + 1
                    entry_price = np.nan
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    index = pd.date_range("2020-01-01", periods=8)
    params = VisibilityParams(profit_target=0.02, max_hold_bars=3)
    entries = pd.DataFrame(
        {"long_entry": [True, False, False, True, False, False, False, False],
         "short_entry": [False] * 8}, index=index,
    )
    data = pd.DataFrame(
        {"open": 100.0, "high": 104.0, "low": 96.0,
         "close": [100.0, 100.0, 103.0, 100.0, 100.0, 100.0, 100.0, 100.0],
         "volume": 1000.0}, index=index,
    )
    result = GeneratedStrategy.generate_signals(None, data, entries, None, params)
    assert result.data["signal"].tolist() == [0, 1, 1, 0, 1, 1, 1, 0]
    entries["short_entry"] = entries["long_entry"]
    entries["long_entry"] = False
    data["close"] = [100.0, 100.0, 97.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    result = GeneratedStrategy.generate_signals(None, data, entries, None, params)
    assert result.data["signal"].tolist() == [0, -1, -1, 0, -1, -1, -1, 0]
    assert result.data["size"].gt(0).all()


if __name__ == "__main__":
    _self_check()

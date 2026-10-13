from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    lookback: int = 9
    min_excursion: float = 0.012
    min_bridges: int = 2
    profit_target: float = 0.04
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791290606"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params):
        if not 5 <= params.lookback <= 10:
            raise ValueError("lookback must be between 5 and 10")
        if not 15 <= params.max_hold <= 20:
            raise ValueError("max_hold must be between 15 and 20")
        if not 0 < params.min_excursion < 1 or params.profit_target <= 0:
            raise ValueError("excursion and target must be positive")
        if params.min_bridges < 1:
            raise ValueError("min_bridges must be positive")
        return int(params.lookback)

    def indicators(self, data, params):
        w = self.warmup_bars(params)
        close = data["close"].astype(float).where(
            np.isfinite(data["close"]) & (data["close"] > 0)
        )
        nodes = [close.shift(w - 1 - k) for k in range(w)]
        valley = w - 3
        out = pd.DataFrame(index=data.index)
        out["valid"] = close.rolling(w, min_periods=w).count().eq(w)
        out["low"] = close.rolling(w, min_periods=w).min()
        out["high"] = close.rolling(w, min_periods=w).max()
        out["pivot"] = nodes[valley]
        left = pd.concat(nodes[:valley], axis=1)
        out["left_high"] = left.max(axis=1, skipna=False)
        out["left_low"] = left.min(axis=1, skipna=False)
        out["drawdown"] = 1.0 - out["pivot"] / out["left_high"]
        out["runup"] = out["pivot"] / out["left_low"] - 1.0
        out["change"] = close.diff()
        out["long_bridges"] = 0
        out["short_bridges"] = 0
        # Horizontal visibility requires all intervening prices below both
        # endpoints; the inverted-price graph requires them above both.
        for i in range(valley):
            for j in range(valley + 1, w):
                interior = pd.concat(nodes[i + 1:j], axis=1)
                roof = interior.max(axis=1, skipna=False)
                floor = interior.min(axis=1, skipna=False)
                out["long_bridges"] += (
                    (nodes[i] > roof) & (nodes[j] > roof)
                ).astype(int)
                out["short_bridges"] += (
                    (nodes[i] < floor) & (nodes[j] < floor)
                ).astype(int)
        out["long_entry"] = (
            out["valid"]
            & out["pivot"].eq(out["low"])
            & out["drawdown"].ge(params.min_excursion)
            & out["long_bridges"].ge(params.min_bridges)
            & out["change"].gt(0)
            & close.lt(out["left_high"])
        )
        out["short_entry"] = (
            out["valid"]
            & out["pivot"].eq(out["high"])
            & out["runup"].ge(params.min_excursion)
            & out["short_bridges"].ge(params.min_bridges)
            & out["change"].lt(0)
            & close.gt(out["left_low"])
        )
        return out

    def generate_signals(self, data, indicators, ctx, params):
        self.warmup_bars(params)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        decisions = np.zeros(len(data), dtype=int)
        position = 0
        entry_price = np.nan
        entry_bar = -1
        for i in range(len(data)):
            if position:
                # The previous close's entry decision fills at this open.
                if entry_bar == -1:
                    entry_price = opens[i]
                    entry_bar = i
                valid_entry = np.isfinite(entry_price) and entry_price > 0
                gain = (
                    position * (closes[i] / entry_price - 1.0)
                    if valid_entry else np.nan
                )
                if (
                    not valid_entry
                    or gain >= params.profit_target
                    or i - entry_bar + 1 >= params.max_hold
                ):
                    position = 0
                    entry_bar = -1
                    entry_price = np.nan
                # Exit decisions remain flat; no same-close reentry.
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                entry_bar = -1
            decisions[i] = position
        df = pd.DataFrame({"signal": decisions, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = Params()
    prices = np.array(
        [110, 109, 108, 107, 106, 104, 100, 102, 105] + [105] * 30,
        dtype=float,
    )
    data = pd.DataFrame(
        {"open": prices, "high": prices + 1, "low": prices - 1,
         "close": prices, "volume": 1000.0},
        index=pd.date_range("2020-01-01", periods=len(prices)),
    )
    ctx = StrategyContext("SPY", "1d", strategy.warmup_bars(params))
    ind = strategy.indicators(data, params)
    assert ind["long_entry"].iloc[8]
    result = strategy.generate_signals(data, ind, ctx, params).data
    assert result["signal"].iloc[8] == 0
    assert result["signal"].iloc[9:27].eq(1).all()
    assert result["signal"].iloc[27] == 0
    rising = data.copy()
    rising.loc[rising.index[10], "close"] = 110.0
    result = strategy.generate_signals(rising, ind, ctx, params).data
    assert result["signal"].iloc[10] == 1
    assert result["signal"].iloc[11] == 0
    mirrored = data.copy()
    for column in ("open", "close"):
        mirrored[column] = 220.0 - data[column]
    mirrored["high"] = mirrored["close"] + 1
    mirrored["low"] = mirrored["close"] - 1
    short_ind = strategy.indicators(mirrored, params)
    assert short_ind["short_entry"].iloc[8]
    result = strategy.generate_signals(mirrored, short_ind, ctx, params).data
    assert result["signal"].iloc[9:27].eq(-1).all()
    assert result["signal"].iloc[27] == 0
    for n in (0, 1, 8, 15):
        prefix = data.iloc[:n]
        prefix_ind = strategy.indicators(prefix, params)
        full_ind = strategy.indicators(data, params).iloc[:n]
        pd.testing.assert_frame_equal(prefix_ind, full_ind)
        prefix_signal = strategy.generate_signals(
            prefix, prefix_ind, ctx, params
        ).data
        full_signal = strategy.generate_signals(
            data, strategy.indicators(data, params), ctx, params
        ).data.iloc[:n]
        pd.testing.assert_frame_equal(prefix_signal, full_signal)


if __name__ == "__main__":
    _self_check()

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class MomentumParams:
    ma_window: int = 20
    breadth_window: int = 6
    entry_z: float = 1.1
    min_breadth: float = 0.25
    atr_window: int = 14
    trail_atr: float = 2.5
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[MomentumParams]):
    strategy_id = "gen_local-w2_1791337535"

    @classmethod
    def params_type(cls):
        return MomentumParams

    def warmup_bars(self, params):
        return int(max(200, params.ma_window + 1,
                       params.breadth_window + 1, params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.ma_window, params.breadth_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if params.entry_z <= 0 or params.trail_atr <= 0 or params.max_hold < 1:
            raise ValueError("Entry, stop and holding parameters must be positive.")
        if not 0 <= params.min_breadth <= 1:
            raise ValueError("min_breadth must be between zero and one.")
        close = data["close"]
        out = pd.DataFrame(index=data.index)
        out["ma"] = close.rolling(params.ma_window).mean()
        out["std"] = close.rolling(params.ma_window).std(ddof=0)
        out["z"] = (close - out["ma"]) / out["std"].replace(0, np.nan)
        out["regime_ma"] = close.rolling(200).mean()
        out["change"] = close.diff()
        out["net_change"] = out["change"].rolling(params.breadth_window).sum()
        out["path_length"] = out["change"].abs().rolling(params.breadth_window).sum()
        out["largest_move"] = out["change"].abs().rolling(params.breadth_window).max()
        # Conservative breadth: remove the largest move as if it helped the trend.
        out["breadth"] = (
            (out["net_change"].abs() - out["largest_move"])
            / (out["path_length"] - out["largest_move"]).replace(0, np.nan)
        ).clip(-1, 1)
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        out["atr"] = out["true_range"].rolling(params.atr_window).mean()
        out["previous_z"] = out["z"].shift(1)
        qualified = out["breadth"].ge(params.min_breadth) & out["atr"].gt(0)
        out["long_entry"] = (
            qualified & close.gt(out["regime_ma"]) & out["net_change"].gt(0)
            & out["z"].ge(params.entry_z) & out["previous_z"].lt(params.entry_z)
        ).fillna(False)
        out["short_entry"] = (
            qualified & close.lt(out["regime_ma"]) & out["net_change"].lt(0)
            & out["z"].le(-params.entry_z) & out["previous_z"].gt(-params.entry_z)
        ).fillna(False)
        return out

    def generate_signals(self, data, indicators, ctx, params):
        df = pd.DataFrame({"signal": 0, "size": 1.0}, index=data.index)
        position, age = 0, 0
        extreme, stop = np.nan, np.nan
        values = indicators[["atr", "regime_ma", "long_entry", "short_entry"]].to_numpy()
        signals = np.zeros(len(data), dtype=int)
        for i, (close, row) in enumerate(zip(data["close"].to_numpy(), values)):
            atr, regime_ma, enter_long, enter_short = row
            if not np.isfinite([close, atr, regime_ma]).all() or atr <= 0:
                position, age = 0, 0
                extreme, stop = np.nan, np.nan
                continue
            if position:
                # First active bar is the fill bar; exclude the pre-entry close.
                age += 1
                if position == 1:
                    extreme = close if age == 1 else max(extreme, close)
                    candidate = extreme - params.trail_atr * atr
                    stop = candidate if age == 1 else max(stop, candidate)
                    stopped = close <= stop
                    regime_lost = close <= regime_ma
                else:
                    # Mirror the high-water trailing rule for short positions.
                    extreme = close if age == 1 else min(extreme, close)
                    candidate = extreme + params.trail_atr * atr
                    stop = candidate if age == 1 else min(stop, candidate)
                    stopped = close >= stop
                    regime_lost = close >= regime_ma
                if stopped or regime_lost or age >= params.max_hold:
                    position, age = 0, 0
                    extreme, stop = np.nan, np.nan
            elif enter_long or enter_short:
                position = 1 if enter_long else -1
                age = 0
                extreme, stop = np.nan, np.nan
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = MomentumParams()
    index = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({"close": [100, 100, 110, 104, 100, 100, 100, 90, 96, 100]}, index=index)
    indicators = pd.DataFrame({
        "atr": [1, 1, 1, 10, 1, 1, 1, 1, 10, 1],
        "regime_ma": [90] * 5 + [110] * 5,
        "long_entry": [True] + [False] * 9,
        "short_entry": [False] * 5 + [True] + [False] * 4,
    }, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0, 0, -1, -1, -1, 0]
    assert result["size"].gt(0).all()
    assert strategy.warmup_bars(params) >= 200


if __name__ == "__main__":
    _self_check()

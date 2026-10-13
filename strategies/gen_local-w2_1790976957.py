from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    streak_length: int = 4
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1790976957"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        if params.streak_length < 2 or params.hold_bars < 1:
            raise ValueError("streak_length must be >= 2 and hold_bars >= 1")
        return int(params.streak_length + 1)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        opening = data["open"].where(data["open"] > 0)
        closing = data["close"].where(data["close"] > 0)
        out["body"] = np.log(closing / opening).replace([np.inf, -np.inf], np.nan)
        out["gap"] = np.log(opening / closing.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["direction"] = np.sign(out["body"]).fillna(0).astype(int)
        groups = out["direction"].ne(out["direction"].shift(1)).cumsum()
        count = out["direction"].groupby(groups).cumcount().add(1).clip(upper=params.streak_length)
        out["streak"] = count.where(out["direction"].ne(0), 0) * out["direction"]
        out["gap_sum"] = out["gap"].rolling(params.streak_length, min_periods=params.streak_length).sum()
        out["body_sum"] = out["body"].rolling(params.streak_length, min_periods=params.streak_length).sum()
        out["net_move"] = out["gap_sum"] + out["body_sum"]
        out["first_mature"] = out["streak"].abs().eq(params.streak_length) & out["streak"].shift(1).abs().lt(params.streak_length)
        out["long_entry"] = out["first_mature"] & out["streak"].lt(0) & out["gap_sum"].gt(0) & out["net_move"].lt(0)
        out["short_entry"] = out["first_mature"] & out["streak"].gt(0) & out["gap_sum"].lt(0) & out["net_move"].gt(0)
        out["entry"] = np.select([out["long_entry"], out["short_entry"]], [1, -1], default=0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: Params) -> SignalFrame:
        self.warmup_bars(params)
        targets = np.zeros(len(data), dtype=int)
        position = 0
        exit_bar = -1
        for i, entry in enumerate(indicators["entry"].fillna(0).to_numpy(dtype=int)):
            if position and i == exit_bar:
                position = 0
            elif not position and entry:
                position = int(entry)
                exit_bar = i + params.hold_bars
            targets[i] = position
        df = pd.DataFrame({"signal": targets, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def demo():
    params = Params()
    strategy = GeneratedStrategy()
    opening = np.empty(45)
    closing = np.empty(45)
    previous = 100.0
    for i in range(45):
        opening[i] = previous * np.exp(0.003)
        closing[i] = opening[i] * np.exp(-0.01 if 1 <= i <= 4 else 0.001)
        previous = closing[i]
    data = pd.DataFrame({"open": opening, "high": np.maximum(opening, closing), "low": np.minimum(opening, closing), "close": closing, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=45))
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=strategy.warmup_bars(params))
    mirrored = data.assign(open=10000 / data["open"], close=10000 / data["close"], high=10000 / data["low"], low=10000 / data["high"])
    for frame, expected in [(data, 1), (mirrored, -1)]:
        result = strategy.generate_signals(frame, strategy.indicators(frame, params), ctx, params).data
        assert result["signal"].iloc[:5].eq(0).all()
        assert result["signal"].iloc[5:23].eq(expected).all()
        assert result["signal"].iloc[23:].eq(0).all()
        assert result["size"].gt(0).all()


if __name__ == "__main__":
    demo()

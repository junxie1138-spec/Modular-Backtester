from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StaircaseParams:
    min_streak: int = 3
    overlap_cap: float = 0.75
    profit_target: float = 0.025
    max_hold: int = 8
    size: float = 1.0


class GeneratedStrategy(BaseStrategy[StaircaseParams]):
    strategy_id = "gen_local-w2_1791267769"

    @classmethod
    def params_type(cls):
        return StaircaseParams

    def warmup_bars(self, params: StaircaseParams) -> int:
        return params.min_streak + 2

    def indicators(self, data: pd.DataFrame, params: StaircaseParams) -> pd.DataFrame:
        if params.min_streak < 1 or params.max_hold < 1 or params.profit_target <= 0 or params.size <= 0 or not 0 <= params.overlap_cap <= 1:
            raise ValueError("Invalid staircase parameters")
        out = pd.DataFrame(index=data.index)
        out["high_change"] = data["high"].diff()
        out["low_change"] = data["low"].diff()
        out["width"] = (data["high"] - data["low"]).where(data["high"].gt(data["low"]))
        out["close_location"] = ((data["close"] - data["low"]) / out["width"]).clip(0, 1)
        out["overlap"] = (np.minimum(data["high"], data["high"].shift(1)) - np.maximum(data["low"], data["low"].shift(1))).clip(lower=0)
        out["overlap_fraction"] = out["overlap"] / np.minimum(out["width"], out["width"].shift(1))
        out["up_step"] = (out["high_change"].gt(0) & out["low_change"].gt(0) & out["close_location"].ge(0.6) & out["overlap_fraction"].le(params.overlap_cap)).fillna(False)
        out["down_step"] = (out["high_change"].lt(0) & out["low_change"].lt(0) & out["close_location"].le(0.4) & out["overlap_fraction"].le(params.overlap_cap)).fillna(False)
        for direction in ("up", "down"):
            step = out[direction + "_step"]
            out[direction + "_streak"] = step.astype(int).groupby((~step).cumsum()).cumsum().clip(upper=params.min_streak)
        out["setup"] = np.select([out["up_streak"].ge(params.min_streak), out["down_streak"].ge(params.min_streak)], [1, -1], default=0)
        out["confirmed"] = out["setup"].where(out["setup"].eq(out["setup"].shift(1)), 0).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame, ctx: StrategyContext, params: StaircaseParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        candidates = indicators["confirmed"].to_numpy(dtype=int)
        setups = indicators["setup"].to_numpy(dtype=int)
        side, entry_bar, entry_price, armed = 0, -1, np.nan, True
        for i in range(len(data)):
            if setups[i] == 0:
                armed = True
            if side:
                # Record the entry open only when that bar has arrived.
                if i == entry_bar:
                    entry_price = opens[i]
                valid_price = np.isfinite(entry_price) and entry_price > 0 and np.isfinite(closes[i])
                hit_target = valid_price and side * (closes[i] / entry_price - 1.0) >= params.profit_target
                if hit_target or i - entry_bar + 1 >= params.max_hold:
                    side = 0
            elif armed and i >= self.warmup_bars(params) - 1 and candidates[i] != 0:
                side = int(candidates[i])
                entry_bar = i + 1
                entry_price = np.nan
                armed = False
            raw[i] = side
        df = pd.DataFrame({"signal": raw, "size": float(params.size)}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    for direction in (1, -1):
        close = 100.0 + direction * 0.5 * np.arange(20)
        data = pd.DataFrame({"open": close, "high": close + (0.2 if direction == 1 else 0.8), "low": close - (0.8 if direction == 1 else 0.2), "close": close, "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=20))
        for target, exit_bar in ((1.0, 13), (0.009, 8)):
            params = StaircaseParams(profit_target=target)
            signals = strategy.generate_signals(data, strategy.indicators(data, params), ctx, params).data["signal"].to_numpy()
            expected = np.zeros(20, dtype=int)
            expected[5:exit_bar] = direction
            assert np.array_equal(signals, expected)
            prefix = data.iloc[:9]
            partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data["signal"].to_numpy()
            assert np.array_equal(partial, signals[:9])
        flat = data.copy()
        flat[["open", "high", "low", "close"]] = 100.0
        assert strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params).data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()

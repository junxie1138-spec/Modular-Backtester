from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    profit_target: float = 0.03
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1790887513"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return 26

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        a, b, c = data["close"].shift(2), data["close"].shift(1), data["close"]
        valid = a.notna() & b.notna() & c.notna() & (a != b) & (a != c) & (b != c)
        patterns = [(a < b) & (b < c), (a < c) & (c < b),
                    (b < a) & (a < c), (b < c) & (c < a),
                    (c < a) & (a < b), (c < b) & (b < a)]
        # Ties invalidate the window rather than inventing an ordering.
        for k, pattern in enumerate(patterns):
            out[f"frequency_{k}"] = pattern.astype(float).where(valid).rolling(18, min_periods=18).mean()
        frequencies = out[[f"frequency_{k}" for k in range(6)]]
        out["entropy"] = -(frequencies * np.log(frequencies.clip(lower=1e-12))).sum(axis=1, min_count=6) / np.log(6.0)
        out["disequilibrium"] = ((frequencies - 1.0 / 6.0) ** 2).sum(axis=1, min_count=6) * 6.0 / 5.0
        out["complexity"] = out["entropy"] * out["disequilibrium"]
        out["prior_complexity"] = out["complexity"].shift(1)
        out["older_complexity"] = out["complexity"].shift(6)
        out["upper_channel"] = data["high"].rolling(12, min_periods=12).max().shift(1)
        out["lower_channel"] = data["low"].rolling(12, min_periods=12).min().shift(1)
        out["entry_ready"] = (
            (out["prior_complexity"] >= 0.06)
            & (out["prior_complexity"] > out["older_complexity"])
            & out["entropy"].shift(1).between(0.35, 0.85)
        ).fillna(False)
        out["entry_direction"] = np.select(
            [out["entry_ready"] & (c > out["upper_channel"]),
             out["entry_ready"] & (c < out["lower_channel"])],
            [1, -1], default=0
        ).astype(int)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: Params) -> SignalFrame:
        if not np.isfinite(params.profit_target) or params.profit_target <= 0:
            raise ValueError("profit_target must be positive and finite")
        if isinstance(params.hold_bars, bool) or not isinstance(params.hold_bars, int) or params.hold_bars < 1:
            raise ValueError("hold_bars must be a positive integer")
        signals = np.zeros(len(data), dtype=int)
        requests = indicators["entry_direction"].fillna(0).to_numpy(dtype=int)
        opens, closes = data["open"].to_numpy(), data["close"].to_numpy()
        position, entry_bar, entry_price = 0, -1, np.nan
        for i in range(len(data)):
            if position:
                if i == entry_bar:
                    entry_price = opens[i]
                    if not np.isfinite(entry_price) or entry_price <= 0:
                        position = 0
                gain = position * (closes[i] / entry_price - 1.0) if position else 0.0
                if gain >= params.profit_target or i - entry_bar + 1 >= params.hold_bars:
                    position = 0
            elif requests[i] in (-1, 1):
                position = int(requests[i])
                entry_bar = i + 1
                entry_price = np.nan
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    for side in (1, -1):
        for target_hit in (False, True):
            data = pd.DataFrame({"open": [100.0] * 6, "close": [100.0] * 6},
                                index=pd.date_range("2020-01-01", periods=6))
            if target_hit:
                data.loc[data.index[2], "close"] = 100.0 * (1.0 + side * 0.04)
            indicators = pd.DataFrame({"entry_direction": [side, 0, 0, 0, 0, 0]}, index=data.index)
            result = strategy.generate_signals(data, indicators, None, Params(0.03, 3)).data
            expected = [0, side, side, 0, 0, 0] if target_hit else [0, side, side, side, 0, 0]
            assert result["signal"].tolist() == expected
            assert (result["size"] > 0).all()


if __name__ == "__main__":
    _self_check()

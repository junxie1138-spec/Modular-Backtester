from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    entropy_window: int = 48
    anchor_window: int = 40
    entropy_floor: float = 0.85
    stretch_threshold: float = 1.0
    hold_bars: int = 18


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local_1791261163"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return int(max(params.entropy_window + 4, params.anchor_window))

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if min(params.entropy_window, params.anchor_window) < 6:
            raise ValueError("Lookbacks must be at least six bars.")
        if not 0.0 <= params.entropy_floor <= 1.0:
            raise ValueError("Entropy floor must be between zero and one.")
        if not np.isfinite(params.stretch_threshold) or params.stretch_threshold <= 0:
            raise ValueError("Stretch threshold must be positive and finite.")
        out = pd.DataFrame(index=data.index)
        out["close"] = data["close"].where(np.isfinite(data["close"]))
        out["lag4"] = out["close"].shift(4)
        out["lag2"] = out["close"].shift(2)
        a, b, c = out["lag4"], out["lag2"], out["close"]
        patterns = [
            (a < b) & (b < c), (a < c) & (c < b),
            (b < a) & (a < c), (b < c) & (c < a),
            (c < a) & (a < b), (c < b) & (b < a),
        ]
        out["pattern"] = np.select(patterns, np.arange(6), default=np.nan)
        valid = out["pattern"].notna()
        entropy = pd.Series(0.0, index=data.index)
        for k in range(6):
            probability = out["pattern"].eq(k).astype(float).where(valid).rolling(
                params.entropy_window, min_periods=params.entropy_window
            ).mean()
            out[f"p{k}"] = probability
            entropy = entropy - probability * np.log(probability.where(probability > 0, 1.0))
        out["entropy"] = entropy / np.log(6.0)
        out["ordinal_vote"] = np.select(
            [(b < a) & (b < c), (b > a) & (b > c)],
            [1, -1], default=0
        )
        out.loc[~valid, "ordinal_vote"] = 0
        out["typical"] = (data["high"] + data["low"] + out["close"]) / 3.0
        out["volume"] = data["volume"].where(
            np.isfinite(data["volume"]) & (data["volume"] >= 0)
        )
        out["weighted_price"] = out["typical"] * out["volume"]
        out["volume_sum"] = out["volume"].rolling(params.anchor_window).sum()
        out["weighted_sum"] = out["weighted_price"].rolling(params.anchor_window).sum()
        out["anchor"] = out["weighted_sum"] / out["volume_sum"].where(out["volume_sum"] > 0)
        out["scale"] = out["close"].rolling(params.anchor_window).std(ddof=0)
        out["stretch"] = (out["close"] - out["anchor"]) / out["scale"].where(out["scale"] > 0)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        if params.hold_bars < 1:
            raise ValueError("Holding period must be positive.")
        complex_path = indicators["entropy"].ge(params.entropy_floor)
        long_entry = complex_path & indicators["ordinal_vote"].eq(1) & indicators["stretch"].le(-params.stretch_threshold)
        short_entry = complex_path & indicators["ordinal_vote"].eq(-1) & indicators["stretch"].ge(params.stretch_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        positions = np.zeros(len(data), dtype=int)
        side, entry_bar = 0, -1
        for i, candidate in enumerate(entries):
            if side and i - entry_bar >= params.hold_bars:
                side = 0
                continue
            if not side and candidate:
                side, entry_bar = int(candidate), i
            positions[i] = side
        df = pd.DataFrame({"signal": positions, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy, params = GeneratedStrategy(), Params()
    index = pd.date_range("2020-01-01", periods=25)
    data = pd.DataFrame(index=index)
    forced = pd.DataFrame(
        {"entropy": 1.0, "ordinal_vote": -1, "stretch": 2.0}, index=index
    )
    forced.iloc[0] = [1.0, 1, -2.0]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=0)
    result = strategy.generate_signals(data, forced, ctx, params).data
    assert result["signal"].iloc[0] == 0
    assert result["signal"].iloc[1:19].eq(1).all()
    assert result["signal"].iloc[19] == 0
    assert result["signal"].iloc[20] == -1
    assert result["size"].gt(0).all()
    index = pd.date_range("2020-01-01", periods=120)
    close = 100.0 + np.sin(np.arange(120) * 1.7) + np.arange(120) * 0.02
    prices = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1000.0}, index=index
    )
    full = strategy.indicators(prices, params)
    assert full.iloc[:strategy.warmup_bars(params) - 1]["entropy"].isna().all()
    assert full.iloc[:80].equals(strategy.indicators(prices.iloc[:80], params))
    flat = prices.assign(open=100.0, high=100.0, low=100.0, close=100.0, volume=0.0)
    flat_signals = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params)
    assert flat_signals.data["signal"].eq(0).all()


if __name__ == "__main__":
    _self_check()

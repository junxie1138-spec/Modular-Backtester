from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    complexity_window: int = 18
    percentile_window: int = 126
    jump_percentile: float = 0.85
    breakout_window: int = 15
    profit_target: float = 0.025
    max_hold: int = 8


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1790978745"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        return int(max(params.breakout_window + 1,
                       params.complexity_window + params.percentile_window + 3))

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        out = pd.DataFrame(index=data.index)
        a, b, c = data["close"].shift(2), data["close"].shift(1), data["close"]
        valid = a.notna() & b.notna() & c.notna()
        # Equal prices retain chronological order, giving deterministic motifs.
        code = 4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)
        out["ordinal_code"] = code.where(valid)
        terms = []
        for motif in (0, 1, 3, 4, 6, 7):
            probability = (code == motif).astype(float).where(valid).rolling(
                params.complexity_window
            ).mean()
            out[f"motif_probability_{motif}"] = probability
            terms.append(-probability * np.log(probability.clip(lower=1e-12)))
        out["entropy"] = pd.concat(terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)
        out["entropy_jump"] = out["entropy"].diff()
        out["jump_threshold"] = out["entropy_jump"].shift(1).rolling(
            params.percentile_window
        ).quantile(params.jump_percentile)
        out["upper_channel"] = data["high"].shift(1).rolling(params.breakout_window).max()
        out["lower_channel"] = data["low"].shift(1).rolling(params.breakout_window).min()
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        ready = indicators[[
            "entropy_jump", "jump_threshold", "upper_channel", "lower_channel"
        ]].notna().all(axis=1)
        discovery = ready & (indicators["entropy_jump"] > 0) & (
            indicators["entropy_jump"] > indicators["jump_threshold"]
        )
        entries = np.select(
            [discovery & (data["close"] > indicators["upper_channel"]),
             discovery & (data["close"] < indicators["lower_channel"])],
            [1, -1], default=0,
        )
        raw = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        side, bars_held = 0, 0
        entry_price = np.nan
        for i in range(len(data)):
            if side:
                # The pending order fills at this bar's open, observed by its close.
                if bars_held == 0:
                    entry_price = opens[i]
                bars_held += 1
                invalid_entry = not np.isfinite(entry_price) or entry_price <= 0
                gain = np.nan if invalid_entry else side * (closes[i] / entry_price - 1.0)
                if invalid_entry or gain >= params.profit_target or bars_held >= params.max_hold:
                    side, bars_held = 0, 0
                    entry_price = np.nan
                raw[i] = side
                # An exit cannot also open a new position on the same close.
                continue
            if entries[i] and np.isfinite(closes[i]) and closes[i] > 0:
                side = int(entries[i])
                bars_held = 0
            raw[i] = side
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = GeneratedParams(max_hold=2)
    index = pd.date_range("2020-01-01", periods=9)
    close = np.array([100., 100., 103., 100., 100., 100., 100., 100., 100.])
    data = pd.DataFrame({
        "open": 100., "high": np.maximum(close, 100.) + 1.,
        "low": np.minimum(close, 100.) - 1., "close": close, "volume": 1000.,
    }, index=index)
    test_indicators = pd.DataFrame({
        "entropy_jump": [1., 0., 0., 0., 1., 0., 0., 0., 0.],
        "jump_threshold": 0.5,
        "upper_channel": [99.] + [200.] * 8,
        "lower_channel": [0.] * 4 + [101.] + [0.] * 4,
    }, index=index)
    signals = strategy.generate_signals(data, test_indicators, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0, -1, -1, 0, 0]
    assert signals["size"].gt(0).all()
    prices = 100. * np.exp(np.cumsum(np.random.default_rng(7).normal(0., 0.01, 400)))
    history = pd.DataFrame({
        "open": prices, "high": prices * 1.01, "low": prices * 0.99,
        "close": prices, "volume": 1000.,
    }, index=pd.date_range("2020-01-01", periods=400))
    full = strategy.indicators(history, params)
    pd.testing.assert_frame_equal(full.iloc[:250], strategy.indicators(history.iloc[:250], params))
    assert full["jump_threshold"].first_valid_index() == history.index[strategy.warmup_bars(params) - 1]

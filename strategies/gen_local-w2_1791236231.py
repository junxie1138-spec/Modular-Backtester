from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class Params:
    rank_window: int = 63
    atr_multiple: float = 2.5


class GeneratedStrategy(BaseStrategy[Params]):
    strategy_id = "gen_local-w2_1791236231"

    @classmethod
    def params_type(cls):
        return Params

    def warmup_bars(self, params: Params) -> int:
        return max(params.rank_window + 5, 15)

    def indicators(self, data: pd.DataFrame, params: Params) -> pd.DataFrame:
        if params.rank_window < 10 or not np.isfinite(params.atr_multiple) or params.atr_multiple <= 0:
            raise ValueError("rank_window must be >= 10 and atr_multiple must be positive")
        out = pd.DataFrame(index=data.index)
        previous_close = data["close"].shift(1)
        out["true_range"] = pd.concat(
            [data["high"] - data["low"],
             (data["high"] - previous_close).abs(),
             (data["low"] - previous_close).abs()], axis=1
        ).max(axis=1)
        out["atr"] = out["true_range"].rolling(14, min_periods=14).mean()
        out["intersection"] = (
            np.minimum(data["high"], data["high"].shift(1))
            - np.maximum(data["low"], data["low"].shift(1))
        ).clip(lower=0)
        out["union"] = (
            np.maximum(data["high"], data["high"].shift(1))
            - np.minimum(data["low"], data["low"].shift(1))
        )
        out["overlap"] = out["intersection"] / out["union"].replace(0, np.nan)
        out["congestion"] = out["overlap"].rolling(5, min_periods=5).mean()
        out["regime_rank"] = out["congestion"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        out["move"] = data["close"] / data["close"].shift(5).replace(0, np.nan) - 1.0
        out["move_rank"] = out["move"].rolling(
            params.rank_window, min_periods=params.rank_window
        ).rank(pct=True)
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: Params
    ) -> SignalFrame:
        signals = np.zeros(len(data), dtype=int)
        position = 0
        age = 0
        high_water = low_water = np.nan
        floor = -np.inf
        ceiling = np.inf
        values = np.column_stack([
            data["close"].to_numpy(), indicators["atr"].to_numpy(),
            indicators["regime_rank"].to_numpy(), indicators["move_rank"].to_numpy()
        ])
        for i, (close, atr, regime, rank) in enumerate(values):
            if not np.isfinite(values[i]).all() or atr <= 0:
                position = 0
                age = 0
                continue
            if position:
                # First held close follows the entry decision by one bar.
                if age == 0:
                    high_water = low_water = close
                high_water = max(high_water, close)
                low_water = min(low_water, close)
                floor = max(floor, high_water - params.atr_multiple * atr)
                ceiling = min(ceiling, low_water + params.atr_multiple * atr)
                age += 1
                # Required high-close stop applies to both directions.
                # Shorts also have a mirrored adverse-price ratchet.
                if close <= floor or (position == -1 and close >= ceiling) or age >= 10:
                    position = 0
                    age = 0
            else:
                impulse = 1 if rank >= 0.85 else (-1 if rank <= 0.15 else 0)
                if regime <= 0.30:
                    position = impulse
                elif regime >= 0.70:
                    position = -impulse
                if position:
                    age = 0
                    floor = -np.inf
                    ceiling = np.inf
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = Params(atr_multiple=2.0)
    index = pd.date_range("2020-01-01", periods=6)
    bars = pd.DataFrame({"close": [100., 100., 104., 103., 101., 101.]}, index=index)
    derived = pd.DataFrame({
        "atr": [1., 1., 1., 10., 1., 1.],
        "regime_rank": 0.1, "move_rank": 0.99
    }, index=index)
    result = strategy.generate_signals(bars, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 1, 0]
    derived["regime_rank"] = 0.9
    assert strategy.generate_signals(bars, derived, None, params).data["signal"].iloc[1] == -1

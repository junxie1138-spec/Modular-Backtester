from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class GeneratedParams:
    pattern_window: int = 6
    max_entropy: float = 0.65
    min_efficiency: float = 0.45
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[GeneratedParams]):
    strategy_id = "gen_local_1791322710"

    @classmethod
    def params_type(cls):
        return GeneratedParams

    def warmup_bars(self, params: GeneratedParams) -> int:
        if not 4 <= params.pattern_window <= 7:
            raise ValueError("pattern_window must be between 4 and 7")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5")
        if not 0.0 < params.max_entropy <= 1.0:
            raise ValueError("max_entropy must be in (0, 1]")
        if not 0.0 < params.min_efficiency <= 1.0:
            raise ValueError("min_efficiency must be in (0, 1]")
        # Three closes per motif, then a rolling distribution and one lag.
        return int(params.pattern_window + 3)

    def indicators(self, data: pd.DataFrame, params: GeneratedParams) -> pd.DataFrame:
        self.warmup_bars(params)
        out = pd.DataFrame(index=data.index)
        close = data["close"].astype(float).replace([np.inf, -np.inf], np.nan)
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Pairwise comparisons encode six permutations; ties favor older closes.
        out["ordinal_code"] = (
            4 * (a > b).astype(int)
            + 2 * (a > c).astype(int)
            + (b > c).astype(int)
        ).where(valid)
        entropy = pd.Series(0.0, index=data.index)
        for code in (0, 1, 3, 4, 6, 7):
            frequency = (
                out["ordinal_code"].eq(code).astype(float).where(valid)
                .rolling(params.pattern_window).mean()
            )
            out[f"frequency_{code}"] = frequency
            entropy -= frequency * np.log(frequency.clip(lower=1e-12)) / np.log(6.0)
        out["entropy"] = entropy.clip(0.0, 1.0)
        out["previous_entropy"] = out["entropy"].shift(1)
        out["change"] = close.diff()
        out["net_change"] = out["change"].rolling(params.pattern_window).sum()
        out["total_movement"] = out["change"].abs().rolling(params.pattern_window).sum()
        out["efficiency"] = (
            out["net_change"].abs() / out["total_movement"].replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        out["direction"] = np.sign(out["net_change"])
        return out

    def generate_signals(
        self,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        ctx: StrategyContext,
        params: GeneratedParams,
    ) -> SignalFrame:
        warmup = self.warmup_bars(params)
        eligible = (
            indicators["entropy"].le(params.max_entropy)
            & indicators["entropy"].lt(indicators["previous_entropy"])
            & indicators["efficiency"].ge(params.min_efficiency)
        ).fillna(False)
        candidates = indicators["direction"].where(eligible, 0).fillna(0).astype(int).to_numpy()
        raw = np.zeros(len(data), dtype=int)
        position = 0
        entry_bar = -1
        for i in range(len(data)):
            if position:
                if i - entry_bar == params.hold_bars:
                    position = 0
            elif i >= warmup - 1 and candidates[i] != 0:
                position = int(candidates[i])
                entry_bar = i
            raw[i] = position
        df = pd.DataFrame(index=data.index)
        df["signal"] = raw
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    params = GeneratedParams()
    strategy = GeneratedStrategy()
    index = pd.date_range("2020-01-01", periods=20)
    close = pd.Series(np.arange(100.0, 120.0), index=index)
    data = pd.DataFrame({"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0})
    derived = strategy.indicators(data, params)
    assert derived["previous_entropy"].first_valid_index() == index[8]
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=9)
    for direction in (-1, 1):
        synthetic = pd.DataFrame({"entropy": 0.2, "previous_entropy": 0.4, "efficiency": 0.8, "direction": direction}, index=index)
        for hold in (3, 4, 5):
            params.hold_bars = hold
            signals = strategy.generate_signals(data, synthetic, ctx, params).data["signal"]
            assert signals.iloc[:9].eq(0).all()
            assert signals.iloc[9:9 + hold].eq(direction).all()
            assert signals.iloc[9 + hold] == 0
    params.hold_bars = 4
    prefix = strategy.indicators(data.iloc[:12], params)
    pd.testing.assert_frame_equal(derived.iloc[:12], prefix)


if __name__ == "__main__":
    _self_check()

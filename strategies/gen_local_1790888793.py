from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalBranchParams:
    complexity_window: int = 24
    complexity_min: float = 0.55
    strength_window: int = 14
    strength_min: float = 0.12
    atr_window: int = 14
    breakeven_pct: float = 0.025
    trail_atr: float = 3.0
    max_hold: int = 18


class GeneratedStrategy(BaseStrategy[OrdinalBranchParams]):
    strategy_id = "gen_local_1790888793"

    @classmethod
    def params_type(cls):
        return OrdinalBranchParams

    def warmup_bars(self, params: OrdinalBranchParams) -> int:
        return max(params.complexity_window + 3, params.strength_window + 1, params.atr_window + 1)

    def indicators(self, data: pd.DataFrame, params: OrdinalBranchParams) -> pd.DataFrame:
        if min(params.complexity_window, params.strength_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if params.trail_atr <= 0 or params.breakeven_pct <= 0 or params.max_hold < 1:
            raise ValueError("Exit parameters must be positive.")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        a, b, c = close.shift(2), close.shift(1), close
        valid = a.notna() & b.notna() & c.notna()
        # Strict comparisons break equal-price ties in chronological order.
        pattern = (4 * (a > b).astype(int) + 2 * (a > c).astype(int) + (b > c).astype(int)).where(valid)
        out["pattern"] = pattern
        previous = pattern.shift(1)
        valid_pair = pattern.notna() & previous.notna()
        joint_entropy = pd.Series(0.0, index=data.index)
        source_entropy = pd.Series(0.0, index=data.index)
        codes = (0, 1, 3, 4, 6, 7)
        for source in codes:
            marginal = pd.Series(0.0, index=data.index)
            for target in codes:
                frequency = ((previous == source) & (pattern == target)).astype(float).where(valid_pair)
                probability = frequency.rolling(params.complexity_window, min_periods=params.complexity_window).mean()
                marginal = marginal + probability
                joint_entropy = joint_entropy - probability * np.log(probability.clip(lower=1e-12))
            source_entropy = source_entropy - marginal * np.log(marginal.clip(lower=1e-12))
        out["branching_complexity"] = ((joint_entropy - source_entropy) / np.log(3.0)).clip(0.0, 1.0)
        up = data["high"].diff()
        down = -data["low"].diff()
        plus = up.where((up > down) & (up > 0), 0.0).where(up.notna() & down.notna())
        minus = down.where((down > up) & (down > 0), 0.0).where(up.notna() & down.notna())
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - close.shift(1)).abs(),
            (data["low"] - close.shift(1)).abs(),
        ], axis=1).max(axis=1).where(close.shift(1).notna())
        out["atr"] = true_range.rolling(params.atr_window, min_periods=params.atr_window).mean()
        denominator = true_range.rolling(params.strength_window, min_periods=params.strength_window).sum().replace(0.0, np.nan)
        out["directional_strength"] = (plus - minus).rolling(params.strength_window, min_periods=params.strength_window).sum() / denominator
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: OrdinalBranchParams) -> SignalFrame:
        qualified = (
            (indicators["branching_complexity"] >= params.complexity_min)
            & (indicators["directional_strength"] >= params.strength_min)
            & (indicators["atr"] > 0)
        ).fillna(False).to_numpy(dtype=bool)
        close = data["close"].to_numpy(dtype=float)
        opening = data["open"].to_numpy(dtype=float)
        low = data["low"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        signal = np.zeros(len(data), dtype=int)
        active = False
        entry = np.nan
        stop = np.nan
        peak = np.nan
        age = 0
        breakeven = False
        for i in range(len(data)):
            if active:
                # Initialize on the fill bar, never read a future opening price.
                if np.isnan(entry):
                    entry = opening[i]
                    stop = entry - params.trail_atr * atr[i - 1]
                    peak = entry
                age += 1
                if not np.isfinite([entry, stop, close[i], low[i], atr[i]]).all() or low[i] <= stop or age >= params.max_hold:
                    active = False
                    continue
                peak = max(peak, close[i])
                if close[i] >= entry * (1.0 + params.breakeven_pct):
                    breakeven = True
                if breakeven:
                    # Close-based peaks avoid assuming intrabar high/low ordering.
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                signal[i] = 1
            elif i >= self.warmup_bars(params) - 1 and qualified[i]:
                active = True
                entry = np.nan
                age = 0
                breakeven = False
                signal[i] = 1
        df = pd.DataFrame({"signal": signal, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    strategy = GeneratedStrategy()
    params = OrdinalBranchParams(complexity_window=2, strength_window=2, atr_window=2)
    index = pd.date_range("2020-01-01", periods=32)
    close = np.full(32, 100.0)
    close[5:9] = [103.0, 105.0, 104.0, 101.0]
    data = pd.DataFrame({"open": 100.0, "high": close + 1.0, "low": close - 0.5,
                         "close": close, "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"branching_complexity": 1.0, "directional_strength": 1.0, "atr": 1.0}, index=index)
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=5)
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[:5].eq(0).all()
    assert result["signal"].iloc[5:9].eq(1).all() and result["signal"].iloc[9] == 0
    params.max_hold = 3
    result = strategy.generate_signals(data, indicators, ctx, params).data
    assert result["signal"].iloc[5:8].eq(1).all() and result["signal"].iloc[8] == 0
    derived = strategy.indicators(data, params)
    assert derived["branching_complexity"].dropna().between(0.0, 1.0).all()
    full = strategy.generate_signals(data, derived, ctx, params).data
    prefix = data.iloc[:20]
    partial = strategy.generate_signals(prefix, strategy.indicators(prefix, params), ctx, params).data
    pd.testing.assert_frame_equal(full.iloc[:20], partial)

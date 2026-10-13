from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class ComplexityParams:
    pattern_window: int = 30
    complexity_drop: float = 0.06
    complexity_ceiling: float = 0.50
    momentum_bars: int = 5
    vol_window: int = 20
    target_vol: float = 0.12
    atr_window: int = 14
    trail_atr: float = 2.5
    hold_bars: int = 8


class GeneratedStrategy(BaseStrategy[ComplexityParams]):
    strategy_id = "gen_local-w2_1791338506"

    @classmethod
    def params_type(cls):
        return ComplexityParams

    def warmup_bars(self, params):
        return int(max(200, params.pattern_window + 9, params.vol_window + 1,
                       params.atr_window + 1, params.momentum_bars + 1))

    def indicators(self, data, params):
        if min(params.pattern_window, params.momentum_bars, params.vol_window,
               params.atr_window, params.hold_bars) < 2:
            raise ValueError("Window and holding parameters must be at least two.")
        if not (0 < params.complexity_ceiling <= 1 and
                0 < params.complexity_drop < 1 and
                params.target_vol > 0 and params.trail_atr > 0):
            raise ValueError("Invalid complexity, volatility or stop parameter.")
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        ret = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        out["return"] = ret
        # Stable ordinal ranks: earlier observations win exact ties.
        a, b, c = ret.shift(2), ret.shift(1), ret
        ranks = np.column_stack([
            (a > b).astype(int) + (a > c).astype(int),
            (b >= a).astype(int) + (b > c).astype(int),
            (c >= a).astype(int) + (c >= b).astype(int),
        ])
        codes = ranks[:, 0] * 9 + ranks[:, 1] * 3 + ranks[:, 2]
        ordinal = pd.Series(codes, index=data.index, dtype=float)
        ordinal = ordinal.where(a.notna() & b.notna() & c.notna())
        out["ordinal"] = ordinal
        # Six permutations; entropy of transitions minus source entropy
        # measures uncertainty about the next pattern, not repertoire breadth.
        states = (5, 7, 11, 15, 19, 21)
        previous = ordinal.shift(1)
        valid_pair = previous.notna() & ordinal.notna()
        source_entropy = pd.Series(0.0, index=data.index)
        joint_entropy = pd.Series(0.0, index=data.index)
        for source in states:
            probability = previous.eq(source).astype(float).where(valid_pair)
            probability = probability.rolling(params.pattern_window).mean()
            out[f"source_probability_{source}"] = probability
            source_entropy -= probability * np.log(probability.clip(lower=1e-12))
            for target in states:
                joint = (previous.eq(source) & ordinal.eq(target)).astype(float)
                joint = joint.where(valid_pair).rolling(params.pattern_window).mean()
                out[f"transition_probability_{source}_{target}"] = joint
                joint_entropy -= joint * np.log(joint.clip(lower=1e-12))
        out["source_entropy"] = source_entropy
        out["joint_entropy"] = joint_entropy
        out["complexity"] = ((joint_entropy - source_entropy) / np.log(6)).clip(0, 1)
        out["complexity_drop"] = out["complexity"].shift(5) - out["complexity"]
        out["ma200"] = close.rolling(200).mean()
        out["volatility"] = ret.rolling(params.vol_window).std(ddof=0) * np.sqrt(252)
        prior_close = close.shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prior_close).abs(),
            (data["low"] - prior_close).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = tr
        out["atr"] = tr.rolling(params.atr_window).mean()
        out["momentum"] = close.pct_change(params.momentum_bars, fill_method=None)
        out["momentum_hurdle"] = (
            out["volatility"] * np.sqrt(params.momentum_bars / 252) * 0.3
        )
        finite = np.isfinite(out[["complexity", "complexity_drop", "ma200",
                                 "volatility", "atr", "momentum"]]).all(axis=1)
        out["ready"] = finite & close.notna() & out["atr"].gt(0) & out["volatility"].gt(0)
        collapse = (out["ready"] & out["complexity"].lt(params.complexity_ceiling)
                    & out["complexity_drop"].ge(params.complexity_drop))
        out["long_entry"] = collapse & close.gt(out["ma200"]) & out["momentum"].gt(out["momentum_hurdle"])
        out["short_entry"] = collapse & close.lt(out["ma200"]) & out["momentum"].lt(-out["momentum_hurdle"])
        out["target_size"] = (params.target_vol / out["volatility"].where(out["volatility"] > 0)).clip(0.05, 0.95).fillna(0.05)
        return out

    def generate_signals(self, data, indicators, ctx: StrategyContext, params):
        df = pd.DataFrame(index=data.index)
        signals = np.zeros(len(data), dtype=int)
        position, age = 0, 0
        high_water = low_water = lower_stop = upper_stop = np.nan
        close = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        ma = indicators["ma200"].to_numpy(dtype=float)
        ready = indicators["ready"].fillna(False).to_numpy(dtype=bool)
        longs = indicators["long_entry"].fillna(False).to_numpy(dtype=bool)
        shorts = indicators["short_entry"].fillna(False).to_numpy(dtype=bool)
        for i in range(len(data)):
            if not ready[i] or not np.isfinite(close[i]):
                position, age = 0, 0
                continue
            if position:
                age += 1
                high_water = max(high_water, close[i])
                low_water = min(low_water, close[i])
                # Required highest-close trail applies to both directions.
                # The short side also has a mirrored adverse-move trail.
                lower_stop = max(lower_stop, high_water - params.trail_atr * atr[i])
                upper_stop = min(upper_stop, low_water + params.trail_atr * atr[i])
                stopped = close[i] <= lower_stop or (position < 0 and close[i] >= upper_stop)
                regime_exit = (position > 0 and close[i] <= ma[i]) or (position < 0 and close[i] >= ma[i])
                if stopped or regime_exit or age >= params.hold_bars:
                    position, age = 0, 0
                    continue
            elif longs[i] or shorts[i]:
                position = 1 if longs[i] else -1
                age = 0
                high_water = low_water = close[i]
                lower_stop = close[i] - params.trail_atr * atr[i]
                upper_stop = close[i] + params.trail_atr * atr[i]
            signals[i] = position
        df["signal"] = signals
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = indicators["target_size"].shift(1).fillna(0.05).astype(float)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = ComplexityParams(trail_atr=2.0)
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"close": [100.0, 104.0, 103.0, 101.0, 101.0]}, index=index)
    derived = pd.DataFrame({
        "atr": [1.0, 1.0, 10.0, 10.0, 10.0], "ma200": 90.0,
        "ready": True, "long_entry": [True, False, False, False, False],
        "short_entry": False, "target_size": [0.2, 0.3, 0.4, 0.5, 0.6],
    }, index=index)
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, 1, 1, 1, 0]
    # ATR expansion cannot loosen the stop ratcheted to 102.
    assert result["size"].tolist() == [0.05, 0.2, 0.3, 0.4, 0.5]
    derived["long_entry"] = False
    derived["short_entry"] = [True, False, False, False, False]
    derived["ma200"] = 110.0
    result = strategy.generate_signals(data, derived, None, params).data
    assert result["signal"].tolist() == [0, -1, 0, 0, 0]


if __name__ == "__main__":
    _self_check()

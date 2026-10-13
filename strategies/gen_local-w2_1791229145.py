from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalParams:
    pattern_window: int = 60
    complexity_min: float = 0.18
    momentum_bars: int = 6
    efficiency_min: float = 0.30
    hold_bars: int = 4


class GeneratedStrategy(BaseStrategy[OrdinalParams]):
    strategy_id = "gen_local-w2_1791229145"

    @classmethod
    def params_type(cls):
        return OrdinalParams

    def warmup_bars(self, params: OrdinalParams) -> int:
        return int(max(200, params.pattern_window + 6, params.momentum_bars + 1))

    def indicators(self, data: pd.DataFrame, params: OrdinalParams) -> pd.DataFrame:
        if params.pattern_window < 12 or params.momentum_bars < 1:
            raise ValueError("Lookbacks must be positive; pattern_window must be at least 12.")
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        if not 0 <= params.complexity_min <= 1 or not 0 <= params.efficiency_min <= 1:
            raise ValueError("Thresholds must be between zero and one.")
        close = data["close"].where(data["close"] > 0)
        ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        a, b, c = ret.shift(2), ret.shift(1), ret
        valid = a.notna() & b.notna() & c.notna()
        # Equal values retain chronological order.
        pattern = (4 * (a > b).astype(int) + 2 * (b > c).astype(int)
                   + (a > c).astype(int)).astype(float).where(valid)
        previous = pattern.shift(3)  # Disjoint return triples avoid shared observations.
        pair_valid = pattern.notna() & previous.notna()
        codes = (0, 2, 3, 4, 5, 7)
        cols = {"return": ret, "pattern": pattern, "previous_pattern": previous}
        for x in codes:
            for y in codes:
                event = ((previous == x) & (pattern == y)).astype(float).where(pair_valid)
                cols[f"p_{x}_{y}"] = event.rolling(
                    params.pattern_window, min_periods=params.pattern_window
                ).mean()
        out = pd.DataFrame(cols, index=data.index)
        source_entropy = pd.Series(0.0, index=data.index)
        target_entropy = pd.Series(0.0, index=data.index)
        joint_entropy = pd.Series(0.0, index=data.index)
        for x in codes:
            source = out[[f"p_{x}_{y}" for y in codes]].sum(axis=1, min_count=6)
            target = out[[f"p_{y}_{x}" for y in codes]].sum(axis=1, min_count=6)
            out[f"source_{x}"] = source
            out[f"target_{x}"] = target
            source_entropy -= source * np.log(source.clip(lower=1e-12))
            target_entropy -= target * np.log(target.clip(lower=1e-12))
            for y in codes:
                p = out[f"p_{x}_{y}"]
                joint_entropy -= p * np.log(p.clip(lower=1e-12))
        out["source_entropy"] = source_entropy
        out["target_entropy"] = target_entropy
        out["joint_entropy"] = joint_entropy
        out["complexity"] = ((source_entropy + target_entropy - joint_entropy)
                             / np.log(6.0)).clip(0.0, 1.0)
        out["diversity"] = target_entropy / np.log(6.0)
        out["drift"] = ret.rolling(params.momentum_bars).sum()
        out["path_length"] = ret.abs().rolling(params.momentum_bars).sum()
        out["efficiency"] = out["drift"].abs() / out["path_length"].replace(0.0, np.nan)
        out["ma200"] = close.rolling(200, min_periods=200).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(
        self, data: pd.DataFrame, indicators: pd.DataFrame,
        ctx: StrategyContext, params: OrdinalParams,
    ) -> SignalFrame:
        if not 3 <= params.hold_bars <= 5:
            raise ValueError("hold_bars must be between 3 and 5.")
        eligible = (
            (indicators["complexity"] >= params.complexity_min)
            & (indicators["diversity"] >= 0.65)
            & (indicators["efficiency"] >= params.efficiency_min)
        )
        long_entry = eligible & (indicators["drift"] > 0) & (data["close"] > indicators["ma200"])
        short_entry = eligible & (indicators["drift"] < 0) & (data["close"] < indicators["ma200"])
        candidates = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        position, entered = 0, -1
        for i, candidate in enumerate(candidates):
            if position and i - entered >= params.hold_bars:
                position = 0
            elif position == 0 and candidate:
                position, entered = int(candidate), i
            signals[i] = position
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    for hold in (3, 4, 5):
        params = OrdinalParams(hold_bars=hold)
        index = pd.date_range("2020-01-01", periods=2 * hold + 3)
        data = pd.DataFrame({"close": 100.0}, index=index)
        direction = np.full(len(index), -1.0)
        direction[0] = 1.0
        indicators = pd.DataFrame({
            "complexity": 1.0, "diversity": 1.0, "efficiency": 1.0,
            "drift": direction, "ma200": 100.0 - direction,
        }, index=index)
        ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
        result = strategy.generate_signals(data, indicators, ctx, params).data
        assert result["signal"].tolist() == [0] + [1] * hold + [0] + [-1] * hold + [0]
        assert (result["size"] > 0).all()
    flat = pd.DataFrame({"close": 100.0}, index=pd.date_range("2020-01-01", periods=250))
    params = OrdinalParams()
    ctx = StrategyContext(symbol="SPY", timeframe="1d", warmup_bars=200)
    result = strategy.generate_signals(flat, strategy.indicators(flat, params), ctx, params)
    assert not result.data["signal"].any()


if __name__ == "__main__":
    _self_check()

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class StrategyParams:
    pattern_window: int = 16
    complexity_edge: float = 0.15


class GeneratedStrategy(BaseStrategy[StrategyParams]):
    strategy_id = "gen_local-w2_1791250035"

    @classmethod
    def params_type(cls) -> type[StrategyParams]:
        return StrategyParams

    def warmup_bars(self, params: StrategyParams) -> int:
        return max(3, int(params.pattern_window)) + 6

    def indicators(self, data: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        window = max(3, int(params.pattern_window))
        out = pd.DataFrame(index=data.index)
        close = data["close"].where(data["close"] > 0)
        opening = data["open"].where(data["open"] > 0)
        out["gap"] = np.log(opening / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        out["body"] = np.log(close / opening).replace([np.inf, -np.inf], np.nan)

        for name in ("gap", "body"):
            series = out[name]
            a, b, c = series.shift(2), series.shift(1), series
            valid = a.notna() & b.notna() & c.notna()
            # Equal values follow chronological order, giving deterministic ranks.
            pattern = ((a > b).astype(int) * 4
                       + (a > c).astype(int) * 2
                       + (b > c).astype(int)).where(valid)
            out[name + "_pattern"] = pattern
            terms = []
            for code in (0, 1, 3, 4, 6, 7):
                probability = pattern.eq(code).astype(float).where(valid).rolling(window, min_periods=window).mean()
                out[f"{name}_prob_{code}"] = probability
                terms.append(-probability * np.log(probability.where(probability > 0, 1.0)))
            out[name + "_entropy"] = pd.concat(terms, axis=1).sum(axis=1, min_count=6) / np.log(6.0)

        out["complexity_difference"] = out["gap_entropy"] - out["body_entropy"]
        out["gap_four"] = out["gap"].rolling(4, min_periods=4).sum()
        out["body_four"] = out["body"].rolling(4, min_periods=4).sum()
        out["repair_balance"] = out["gap_four"] + out["body_four"]
        ready = out[["complexity_difference", "gap_four", "repair_balance"]].notna().all(axis=1)
        setup = (out["complexity_difference"].gt(params.complexity_edge)
                 & out["gap_four"].lt(0)
                 & out["repair_balance"].gt(0))
        out["absorption_setup"] = setup.astype(float).where(ready)
        out["entry_support"] = out["absorption_setup"].rolling(4, min_periods=4).mean()
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: StrategyParams) -> SignalFrame:
        df = pd.DataFrame(index=data.index)
        # Hold precisely while the entry condition remains true; its reversal exits.
        entry_condition = indicators["entry_support"].ge(0.5).fillna(False)
        df["signal"] = entry_condition.astype(int)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        df["size"] = 0.95
        return SignalFrame(data=df, signal_column="signal", size_column="size")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    index = pd.date_range("2020-01-01", periods=96, freq="B")
    gaps = rng.normal(-0.002, 0.005, len(index))
    bodies = rng.normal(0.004, 0.004, len(index))
    closes = 100.0 * np.exp(np.cumsum(gaps + bodies))
    previous = np.concatenate(([100.0], closes[:-1]))
    opens = previous * np.exp(gaps)
    data = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.001,
                         "low": np.minimum(opens, closes) * 0.999,
                         "close": closes, "volume": 1000000.0}, index=index)
    strategy = GeneratedStrategy()
    params = StrategyParams()
    indicators = strategy.indicators(data, params)
    entropy = indicators[["gap_entropy", "body_entropy"]].dropna()
    assert not entropy.empty
    assert ((entropy >= -1e-12) & (entropy <= 1.0 + 1e-12)).all().all()
    pd.testing.assert_frame_equal(strategy.indicators(data.iloc[:64], params), indicators.iloc[:64])
    probe = pd.DataFrame({"entry_support": [np.nan, 0.5, 0.75, 0.25, 0.0]}, index=index[:5])
    signals = strategy.generate_signals(data.iloc[:5], probe, None, params).data
    assert signals["signal"].tolist() == [0, 0, 1, 1, 0]
    assert signals["size"].gt(0).all()

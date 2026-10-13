from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class RangeComplexityParams:
    pattern_window: int = 18
    min_complexity: float = 0.10
    gap_atr: float = 0.60
    min_repair: float = 0.50
    atr_period: int = 14
    activation_pct: float = 0.004
    trail_atr: float = 1.5


class GeneratedStrategy(BaseStrategy[RangeComplexityParams]):
    strategy_id = "gen_local-w2_1791312178"

    @classmethod
    def params_type(cls) -> type[RangeComplexityParams]:
        return RangeComplexityParams

    def warmup_bars(self, params: RangeComplexityParams) -> int:
        return max(params.pattern_window + 2, params.atr_period + 2)

    def indicators(self, data: pd.DataFrame, params: RangeComplexityParams) -> pd.DataFrame:
        if params.pattern_window < 3 or params.atr_period < 1:
            raise ValueError("Lookbacks must be positive; pattern_window must be at least three.")
        if not (0 < params.min_complexity < 1 and params.gap_atr > 0
                and 0 < params.min_repair < 1 and params.activation_pct > 0
                and params.trail_atr > 0):
            raise ValueError("Invalid complexity, gap, repair, or stop parameters.")
        out = pd.DataFrame(index=data.index)
        out["previous_close"] = data["close"].shift(1)
        out["range_fraction"] = (data["high"] - data["low"]) / data["close"].where(data["close"] > 0)
        triples = np.column_stack([out["range_fraction"].shift(2),
                                   out["range_fraction"].shift(1), out["range_fraction"]])
        order = np.argsort(triples, axis=1, kind="stable")
        codes = order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]
        valid = np.isfinite(triples).all(axis=1)
        out["motif"] = np.where(valid, codes, np.nan)
        for code in (5, 7, 11, 15, 19, 21):
            occurrence = out["motif"].eq(code).astype(float).where(valid)
            out[f"p_{code}"] = occurrence.rolling(params.pattern_window, min_periods=params.pattern_window).mean()
        probabilities = out[[f"p_{code}" for code in (5, 7, 11, 15, 19, 21)]]
        entropy = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(axis=1, min_count=6)
        mixture = (probabilities + 1.0 / 6.0) / 2.0
        mixture_entropy = -(mixture * np.log(mixture.clip(lower=1e-12))).sum(axis=1, min_count=6)
        out["ordinal_entropy"] = entropy / np.log(6.0)
        out["ordinal_disequilibrium"] = ((mixture_entropy - 0.5 * entropy - 0.5 * np.log(6.0)) / np.log(2.0)).clip(lower=0)
        out["complexity"] = out["ordinal_entropy"] * out["ordinal_disequilibrium"]
        out["true_range"] = pd.concat([
            data["high"] - data["low"],
            (data["high"] - out["previous_close"]).abs(),
            (data["low"] - out["previous_close"]).abs(),
        ], axis=1).max(axis=1, skipna=False)
        out["atr"] = out["true_range"].rolling(params.atr_period, min_periods=params.atr_period).mean()
        out["previous_atr"] = out["atr"].shift(1)
        out["down_gap"] = out["previous_close"] - data["open"]
        out["gap_atr"] = out["down_gap"] / out["previous_atr"].where(out["previous_atr"] > 0)
        out["repair_fraction"] = (data["close"] - data["open"]) / out["down_gap"].where(out["down_gap"] > 0)
        out["complexity_agrees"] = out["complexity"].ge(params.min_complexity)
        out["gap_repair_agrees"] = out["gap_atr"].ge(params.gap_atr) & out["repair_fraction"].ge(params.min_repair) & out["repair_fraction"].lt(1.0)
        out["entry"] = (out["complexity_agrees"] & out["gap_repair_agrees"] & out["atr"].gt(0)).fillna(False)
        return out

    def generate_signals(self, data: pd.DataFrame, indicators: pd.DataFrame,
                         ctx: StrategyContext, params: RangeComplexityParams) -> SignalFrame:
        raw = np.zeros(len(data), dtype=int)
        active = False
        pending = False
        armed = False
        entry = peak = stop = 0.0
        age = 0
        prices = data[["open", "high", "low", "close"]].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        entries = indicators["entry"].fillna(False).to_numpy(dtype=bool)
        for i, (opening, high, low, close) in enumerate(prices):
            if pending:
                pending = False
                if i > 0 and np.isfinite(opening) and opening > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0:
                    active = True
                    entry = peak = opening
                    stop = entry - params.trail_atr * atr[i - 1]
                    age = 0
                    armed = False
            if active:
                age += 1
                if not np.isfinite([opening, high, low, close, atr[i]]).all() or atr[i] <= 0:
                    active = False
                    continue
                # Test the previously known stop before today's high can raise it.
                if low <= stop:
                    active = False
                    continue
                peak = max(peak, high)
                if peak >= entry * (1.0 + params.activation_pct):
                    armed = True
                if armed:
                    stop = max(stop, entry, peak - params.trail_atr * atr[i])
                # A raised stop is checked at the close, then against future lows.
                # All exit decisions execute next bar through the signal shift.
                if close <= stop or age >= 2:
                    active = False
                    continue
                raw[i] = 1
            elif entries[i] and np.isfinite(prices[i]).all() and close > 0:
                raw[i] = 1
                pending = True
        df = pd.DataFrame({"signal": raw, "size": 1.0}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check() -> None:
    strategy = GeneratedStrategy()
    params = RangeComplexityParams()
    index = pd.date_range("2020-01-01", periods=5)
    data = pd.DataFrame({"open": [100.0] * 5, "high": [100.2] * 5,
                         "low": [99.8] * 5, "close": [100.0] * 5,
                         "volume": [1000.0] * 5}, index=index)
    indicators = pd.DataFrame({"atr": [1.0] * 5,
                               "entry": [True, False, False, False, False]}, index=index)
    signals = strategy.generate_signals(data, indicators, None, params).data
    assert signals["signal"].tolist() == [0, 1, 1, 0, 0]
    data.loc[index[1], ["high", "close"]] = [100.6, 99.9]
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
    data.loc[index[1], ["high", "close"]] = [103.0, 101.0]
    assert strategy.generate_signals(data, indicators, None, params).data["signal"].tolist() == [0, 1, 0, 0, 0]
    indicators["entry"] = False
    assert not strategy.generate_signals(data, indicators, None, params).data["signal"].any()
    derived = strategy.indicators(data, params)
    assert not derived["entry"].any()
    assert derived.index.equals(data.index)
    assert signals["size"].gt(0).all()
    widths = np.resize([1.0, 3.0, 2.0], 60)
    history = pd.DataFrame({"open": 100.0, "high": 100.0 + widths / 2,
                            "low": 100.0 - widths / 2, "close": 100.0,
                            "volume": 1000.0}, index=pd.date_range("2020-01-01", periods=60))
    history.iloc[-1] = [98.0, 100.0, 97.0, 99.2, 1000.0]
    derived = strategy.indicators(history, params)
    assert derived["entry"].iloc[-1]
    assert not strategy.indicators(history, RangeComplexityParams(min_complexity=0.99))["entry"].any()
    assert not strategy.indicators(history, RangeComplexityParams(gap_atr=99.0))["entry"].any()
    full = strategy.generate_signals(history, derived, None, params).data
    for end in (1, 19, 20, 30, 59):
        prefix = history.iloc[:end]
        actual = strategy.generate_signals(prefix, strategy.indicators(prefix, params), None, params).data
        pd.testing.assert_frame_equal(actual, full.iloc[:end])


if __name__ == "__main__":
    _self_check()

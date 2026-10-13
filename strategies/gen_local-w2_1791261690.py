from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.core.types import SignalFrame, StrategyContext
from backtester.strategies.base import BaseStrategy


@dataclass(slots=True)
class OrdinalReleaseParams:
    pattern_window: int = 10
    entropy_drop: float = 0.08
    ordinal_bias: float = 0.20
    body_window: int = 3
    body_threshold: float = 0.15
    atr_window: int = 14
    breakeven_pct: float = 0.004
    trail_atr: float = 1.0
    max_hold: int = 2


class GeneratedStrategy(BaseStrategy[OrdinalReleaseParams]):
    strategy_id = "gen_local-w2_1791261690"

    @classmethod
    def params_type(cls):
        return OrdinalReleaseParams

    def warmup_bars(self, params):
        return int(max(2 * params.pattern_window + 2, params.body_window, params.atr_window + 1))

    def indicators(self, data, params):
        if min(params.pattern_window, params.body_window, params.atr_window) < 2:
            raise ValueError("Indicator windows must be at least two.")
        if params.max_hold not in (1, 2):
            raise ValueError("max_hold must be one or two.")
        if params.trail_atr <= 0 or params.breakeven_pct <= 0:
            raise ValueError("Stop parameters must be positive.")
        if not (0 <= params.entropy_drop <= 1 and 0 <= params.ordinal_bias <= 1 and 0 <= params.body_threshold <= 1):
            raise ValueError("Signal thresholds must lie in [0, 1].")
        out = pd.DataFrame(index=data.index)
        close = data["close"]
        triples = pd.concat([close.shift(2), close.shift(1), close], axis=1)
        ranks = np.argsort(triples.to_numpy(), axis=1, kind="stable")
        code = ranks[:, 0] * 9 + ranks[:, 1] * 3 + ranks[:, 2]
        # Ties follow chronological order; missing triples never become patterns.
        pattern = pd.Series(code, index=data.index, dtype=float).where(triples.notna().all(axis=1))
        out["pattern"] = pattern
        entropy = pd.Series(0.0, index=data.index)
        for code_value in (5, 7, 11, 15, 19, 21):
            frequency = pattern.eq(code_value).astype(float).where(pattern.notna()).rolling(params.pattern_window).mean()
            out["frequency_" + str(code_value)] = frequency
            entropy = entropy - frequency * np.log(frequency.where(frequency > 0, 1.0))
        out["entropy"] = entropy / np.log(6.0)
        out["entropy_drop"] = out["entropy"].shift(params.pattern_window) - out["entropy"]
        out["ordinal_bias"] = out["frequency_5"] - out["frequency_21"]
        body = (data["close"] - data["open"]).rolling(params.body_window).sum()
        span = (data["high"] - data["low"]).rolling(params.body_window).sum()
        out["body_sum"] = body
        out["range_sum"] = span
        out["body_efficiency"] = body / span.where(span > 0)
        previous_close = close.shift(1)
        true_range = pd.concat([
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        out["true_range"] = true_range
        out["atr"] = true_range.rolling(params.atr_window).mean()
        return out.replace([np.inf, -np.inf], np.nan)

    def generate_signals(self, data, indicators, ctx, params):
        ready = indicators[["entropy_drop", "ordinal_bias", "body_efficiency", "atr"]].notna().all(axis=1)
        release = indicators["entropy_drop"] >= params.entropy_drop
        long_entry = ready & release & (indicators["ordinal_bias"] > params.ordinal_bias) & (indicators["body_efficiency"] > params.body_threshold)
        short_entry = ready & release & (indicators["ordinal_bias"] < -params.ordinal_bias) & (indicators["body_efficiency"] < -params.body_threshold)
        entries = np.select([long_entry, short_entry], [1, -1], default=0)
        signals = np.zeros(len(data), dtype=int)
        opens = data["open"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        atr = indicators["atr"].to_numpy(dtype=float)
        position = 0
        entry = 0.0
        age = 0
        best = 0.0
        stop = 0.0
        armed = False
        for i in range(len(data)):
            # Yesterday's decision fills at today's open.
            previous_signal = signals[i - 1] if i else 0
            if position and previous_signal == 0:
                position = 0
            if position == 0 and previous_signal:
                if not (np.isfinite(opens[i]) and opens[i] > 0 and np.isfinite(atr[i - 1]) and atr[i - 1] > 0):
                    signals[i] = 0
                    continue
                position = int(previous_signal)
                entry = opens[i]
                age = 0
                best = 0.0
                stop = -params.trail_atr * atr[i - 1]
                armed = False
            if position:
                age += 1
                if not (np.isfinite(closes[i]) and closes[i] > 0 and np.isfinite(atr[i]) and atr[i] > 0):
                    signals[i] = 0
                    continue
                profit = position * (closes[i] - entry)
                hit = profit <= stop
                best = max(best, profit)
                armed = armed or best >= entry * params.breakeven_pct
                if armed:
                    # Signed profit stop only increases: price stop rises for longs,
                    # falls for shorts. Activate at breakeven, then trail closes.
                    stop = max(stop, 0.0, best - params.trail_atr * atr[i])
                    hit = hit or profit <= stop
                # ponytail: close-observed stops fill next open; intraday stops need a finer execution contract.
                signals[i] = 0 if hit or age >= params.max_hold else position
            else:
                signals[i] = entries[i]
        df = pd.DataFrame({"signal": signals, "size": 0.95}, index=data.index)
        df["signal"] = df["signal"].shift(1).fillna(0).astype(int)
        return SignalFrame(data=df, signal_column="signal", size_column="size")


def _self_check():
    strategy = GeneratedStrategy()
    params = OrdinalReleaseParams()
    index = pd.date_range("2020-01-01", periods=7)
    data = pd.DataFrame({"open": 100.0, "high": 102.0, "low": 98.0, "close": [100.0, 101.0, 99.8, 100.0, 99.0, 100.2, 100.0], "volume": 1000.0}, index=index)
    indicators = pd.DataFrame({"entropy_drop": 0.2, "ordinal_bias": [0.5, 0, 0, -0.5, 0, 0, 0], "body_efficiency": [0.5, 0, 0, -0.5, 0, 0, 0], "atr": 1.0}, index=index)
    actual = strategy.generate_signals(data, indicators, None, params).data
    assert actual["signal"].tolist() == [0, 1, 1, 0, -1, -1, 0]
    params.max_hold = 1
    actual = strategy.generate_signals(data, indicators, None, params).data
    assert actual["signal"].tolist() == [0, 1, 0, 0, -1, 0, 0]
    result = strategy.indicators(data, OrdinalReleaseParams())
    assert result.index.equals(data.index)
    assert not np.isinf(result.to_numpy()).any()


if __name__ == "__main__":
    _self_check()
